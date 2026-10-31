"""Matched-expert APPROXIMATE replay of paper section 3.5 / Algorithm 2."""
from __future__ import annotations

from recognition.common_budget_pipeline import counts, micro
from scoring.paper_image_router import (PaperRouterConfig, assemble_t3,
                                        localized_word_scores, paper_decision)


def crop_transcript(rows):
    # Preserve every expert output, including duplicates and empty readings.
    return assemble_t3([dict(r, instance_id=f'row:{i:06d}') for i, r in enumerate(rows)])


def score_transcript(t1, rows, caption, semantic, tokenizer, config):
    t3 = crop_transcript(rows)
    texts = {'T1': t1, 'T2': caption, 'T3': t3['text']}
    lengths = {k: len(tokenizer(v, add_special_tokens=True, truncation=False)['input_ids'])
               for k, v in texts.items()}
    overflow = any(n > semantic.config.max_sequence_length for n in lengths.values())
    if overflow:
        return dict(texts=texts, token_lengths=lengths, context_overflow=True,
                    decision=None, scores={'paper_selected': None, 'joined_t3': None})
    s1, s3 = semantic.compare_candidates(t1, t3['text'], caption)
    decision = paper_decision(t1, t3['text'], caption, s1, s3, config)
    return dict(texts=texts, token_lengths=lengths, context_overflow=False,
                decision=decision, scores=localized_word_scores(decision, t3))


def accepted_images(scores, keys, *, threshold=None, fallback_images=None):
    if set(scores) != set(keys) or (threshold is None) == (fallback_images is None):
        raise ValueError('Select exactly one explicit threshold or fallback quota')
    import math
    if any(v is not None and not math.isfinite(v) for v in scores.values()):
        raise ValueError('Nonfinite router confidence')
    eligible = [s for s, v in scores.items() if v is not None]
    if threshold is not None:
        return sorted(s for s in eligible if scores[s] >= threshold)
    if not 0 <= fallback_images <= len(scores):
        raise ValueError('Invalid fallback quota')
    keep = len(scores) - fallback_images
    if len(eligible) < keep:
        raise ValueError('Eligible images cannot satisfy the fallback quota')
    return sorted(eligible, key=lambda s: (-scores[s], keys[s], s))[:keep]


def evaluate_routing(features, light_branches, t1_branches, fallback_branches,
                     accepted, gate):
    population = set(features)
    if any(set(v) != population for v in (light_branches, t1_branches, fallback_branches)):
        raise ValueError('Missing branch images; denominators cannot be reduced')
    accepted = set(accepted)
    if not accepted <= population or gate not in ('paper_selected', 'joined_t3'):
        raise ValueError('Invalid gate or accepted images')
    selected = {}; sources = {'T1': 0, 'T3': 0, 'DeepSolo': 0}
    for sid in sorted(population):
        if sid not in accepted:
            selected[sid] = fallback_branches[sid]; sources['DeepSolo'] += 1
        else:
            if features[sid]['scores'][gate] is None:
                raise ValueError('Ineligible image accepted')
            source = features[sid]['decision']['semantic_winner'] if gate == 'paper_selected' else 'T3'
            selected[sid] = t1_branches[sid] if source == 'T1' else light_branches[sid]
            sources[source] += 1
    n = len(population); fallback = sources['DeepSolo']
    return dict(combined=micro(selected.values()), source_counts=sources,
                accepted_ids=sorted(accepted), per_image=selected,
                deepsolo_images=fallback, total_images=n,
                deepsolo_percentage=100 * fallback / n if n else None,
                scope='logical cached routing; no measured online inference latency')


def whole_image_counts(text, truth):
    # Native PARSeq T1 is one sequence, without invented boxes or word splitting.
    return counts([{'raw_text': text}], truth, [False])
