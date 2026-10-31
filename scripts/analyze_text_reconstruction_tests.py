"""Post-hoc error analysis of completed, immutable frozen-OCR experiments."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from recognition.scene_text_datasets import parse_totaltext_annotation


def read(path):return json.loads(Path(path).read_text())


def text_tokens(texts):return [s for text in texts for s in (text.casefold().split() or [''])]


def change_kind(before,after):
    left=''.join(before.casefold().split());right=''.join(after.casefold().split())
    if left==right:return 'word_boundaries_only'
    if right.startswith(left):return 'suffix_completion'
    if right.endswith(left):return 'prefix_completion'
    return 'internal_or_multiple_completion'


def analyze(root,draws):
    result=read(root/'results.json');protocol=read(root/'protocol.json');data=read(root/'observations.json')
    output={}
    for phase in ['evaluation','test']:
        rows=read(root/(phase+'_per_image.json'));samples={s['sample_id']:s for s in data[phase]}
        arm='denoiser_preserve';changed=[];kinds=Counter();reasons=Counter();pool_present=pool_absent=0
        subset_totals={key:Counter() for key in ['word_boundaries_only','letter_completions_only']}
        cer={name:Counter() for name in ['baseline','denoiser','denoiser_preserve']}
        for row in rows:
            sample=samples[row['sample_id']]
            baseline={tuple(g['parents']):g for g in row['arms']['baseline']['groups']}
            groups=row['arms'][arm]['groups']
            pool={token for b in sample['blocks'] for token in text_tokens([b['text']])}
            pool.update(token for op in sample['operations'] for token in text_tokens(op['texts']))
            for token,count in row['arms'][arm]['recovered_texts'].items():
                if token in pool:pool_present+=count
                else:pool_absent+=count
            gt=[w for w in parse_totaltext_annotation(Path(sample['annotation_path'])) if not w.ignore]
            gt.sort(key=lambda w:((w.bbox_xyxy[1]+w.bbox_xyxy[3])/2,w.bbox_xyxy[0]))
            target=' '.join(w.transcription for w in gt).casefold()
            for name in cer:
                text=' '.join(s for g in row['arms'][name]['groups'] for s in g['texts']).casefold()
                cer[name]['edit_distance']+=Levenshtein.distance(text,target)
                cer[name]['care_gt_characters']+=len(target)
            for group in groups:
                reference=baseline[tuple(group['parents'])]
                before,after=' '.join(reference['texts']),' '.join(group['texts'])
                reasons[group['repair_reason']]+=1
                if before.casefold()==after.casefold():continue
                kind=change_kind(before,after);kinds[kind]+=1
                after_tokens=Counter(text_tokens(group['texts']));before_tokens=Counter(text_tokens(reference['texts']))
                possible_gains=after_tokens & Counter(row['arms'][arm]['recovered_texts'])
                possible_losses=before_tokens & Counter(row['arms'][arm]['damaged_texts'])
                changed.append(dict(sample_id=row['sample_id'],parents=group['parents'],bbox=group['bbox'],
                    before=before,after=after,kind=kind,image_recovered_texts=dict(possible_gains),image_damaged_texts=dict(possible_losses)))
            expected=Counter(w.transcription.casefold() for w in gt)
            old_matched=Counter(text_tokens([s for g in baseline.values() for s in g['texts']])) & expected
            for subset,counts in subset_totals.items():
                values=[]
                for group in groups:
                    reference=baseline[tuple(group['parents'])]
                    before,after=' '.join(reference['texts']),' '.join(group['texts'])
                    boundary=change_kind(before,after)=='word_boundaries_only'
                    take=boundary if subset=='word_boundaries_only' else not boundary
                    values.extend(group['texts'] if take else reference['texts'])
                predicted=Counter(text_tokens(values));matched=predicted & expected
                counts.update(correct=sum(matched.values()),predictions=sum(predicted.values()),gt_words=sum(expected.values()),
                              recovered=sum((matched-old_matched).values()),damaged=sum((old_matched-matched).values()))
        rng=np.random.default_rng(protocol['resolved_config']['seed']);differences=[]
        for _ in range(draws):
            index=rng.integers(0,len(rows),len(rows))
            differences.append(sum(rows[i]['arms'][arm]['correct']-rows[i]['arms']['baseline']['correct'] for i in index))
        sensitivity={}
        clean=[r for r in rows if not r['known_rgb_duplicate']]
        for name in ['baseline','neuspell_insertions','neuspell_long_insertions','structure_raw','denoiser','denoiser_preserve']:
            total={key:sum(r['arms'][name][key] for r in clean) for key in ['correct','predictions','gt_words','recovered','damaged']}
            total['images']=len(clean);total['f1']=2*total['correct']/(total['predictions']+total['gt_words'])
            sensitivity[name]=total
            if phase=='test':
                saved=result['cohorts'][phase]['without_known_rgb_duplicates'][name]
                if any(saved[k]!=v for k,v in total.items() if k in saved):raise RuntimeError('Duplicate sensitivity mismatch')
        for name,counts in cer.items():counts['reading_order_cer']=counts['edit_distance']/counts['care_gt_characters'] if counts['care_gt_characters'] else None
        for counts in subset_totals.values():counts['f1']=2*counts['correct']/(counts['predictions']+counts['gt_words'])
        output[phase]=dict(letter_preserving_changed_groups=len(changed),change_types=dict(kinds),rejection_reasons=dict(reasons),
            recovered_occurrences_with_spelling_in_structural_pool=pool_present,recovered_occurrences_absent_from_structural_pool=pool_absent,
            conditional_correct_count_delta_interval95=np.quantile(differences,[.025,.975]).tolist(),
            images_more_correct=sum(r['arms'][arm]['correct']>r['arms']['baseline']['correct'] for r in rows),
            images_fewer_correct=sum(r['arms'][arm]['correct']<r['arms']['baseline']['correct'] for r in rows),
            without_known_duplicate_images=sensitivity,reading_order_character_diagnostic={k:dict(v) for k,v in cer.items()},
            posthoc_subset_controls={k:dict(v) for k,v in subset_totals.items()},
            changed_groups=changed)
    return dict(reproduction_mode='approximate',claim_label='approximate reproduction; post-hoc saved-output analysis',
        result_sha256=sha256((root/'results.json').read_bytes()).hexdigest(),analysis_code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        source_protocol_sha256=sha256((root/'protocol.json').read_bytes()).hexdigest(),cohorts=output,
        bootstrap_draws=draws,selection_altered=False,
        limitations=[
            'Lexical matches and changes are per-image text multiplicities, not localized GT identities.',
            'Pool membership concerns available spellings, not simultaneous feasible structural choices.',
            'Group examples show overlap with whole-image gains/losses, not disjoint causal attributions.',
            'Reading-order CER is a post-hoc diagnostic, including casefold text, punctuation, spaces and ordering errors; it is not a pure recognizer CER.',
            'Boundary-only and completion-only subset controls are retrospective diagnostics, not calibrated or promoted policies.',
            'Bootstrap intervals condition on fitted models/policies and omit training variability and historical search.',
            'Calibrated policies were fixed before evaluation and remain KEEP; raw endpoints are not promoted.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--bootstrap-draws',type=int,required=True)
    args=parser.parse_args()
    if args.bootstrap_draws<1:raise ValueError('Positive bootstrap budget required')
    report=analyze(args.run,args.bootstrap_draws)
    with args.output.open('x') as stream:json.dump(report,stream,ensure_ascii=False,sort_keys=True,indent=2);stream.write('\n')
    for name,row in report['cohorts'].items():print(name,json.dumps({k:v for k,v in row.items() if k not in ['changed_groups','without_known_duplicate_images']}))


if __name__=='__main__':main()
