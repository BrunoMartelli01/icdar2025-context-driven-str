"""Post-hoc boundary failure analysis; never changes fitted models or policies."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from rapidfuzz.distance import Levenshtein

from recognition.contextual_boundary_selection import edit_label
from recognition.text_reconstruction import structural_oracle
from scripts.run_text_reconstruction_tests import annotations, read, sha, write
from scripts.run_contextual_boundary_selection import evaluate, bootstrap


def analyze(root):
    root=Path(root);protocol=read(root/'protocol.json');c=protocol['resolved_config']
    data=read(root/'candidates.json');results=read(root/'results.json');pred=read(root/'predictions.json')
    selected=read(root/'changes.json');scores=read(root/'scores.json');selection=read(root/'selection_frozen.json')
    external={s.casefold() for s in read(c['paths']['spell_vocabulary'])['labels']}
    primary_seed=c['selector']['seeds'][0];out=dict(analysis_code_sha256=sha(__file__),results_sha256=sha(root/'results.json'),
        selection_sha256=sha(root/'selection_frozen.json'),models_or_policies_changed=False,
        claim_label='approximate reproduction / post-hoc exploratory analysis',cohorts={})
    for phase in ('calibration','evaluation','test'):
        samples=data[phase];by={s['sample_id']:s for s in samples};labels={};pool_stats=Counter();context_stats=Counter();oracle={}
        for s in samples:
            sid=s['sample_id'];words=annotations(s);labels[sid]=[]
            for i,(b,options) in enumerate(zip(s['blocks'],s['candidates'])):
                rows=[]
                if len(options)>1:
                    has_context=any(s['contexts'][i]);context_stats['proposed_blocks']+=1
                    context_stats['proposed_blocks_with_context']+=int(has_context)
                    context_stats['proposed_blocks_known_external']+=int(b['text'].casefold() in external)
                    context_ids=[j for side in s['contexts'][i] for j in side]
                    gt_spellings={w['text'].casefold() for w in words}
                    context_stats['neighbour_occurrences']+=len(context_ids)
                    context_stats['neighbour_exact_lexical_occurrences']+=sum(s['blocks'][j]['text'].casefold() in gt_spellings for j in context_ids)
                for j,o in enumerate(options[1:],1):
                    label=edit_label(s['blocks'],i,o['text'],words)
                    row=dict(candidate_index=j,**label)
                    rows.append(row);pool_stats[{0:'neutral',1:'beneficial',2:'damaging'}[label['label']]]+=1
                    pool_stats['neural_candidates']+=int('byt5' in o['origins'])
                    pool_stats['lexical_candidates']+=int('lexical' in o['origins'])
                    context_stats[('context_' if any(s['contexts'][i]) else 'no_context_')+{0:'neutral',1:'beneficial',2:'damaging'}[label['label']]]+=1
                labels[sid].append(rows)
        # Candidate availability diagnostic, not an inference selection rule.
        for origin in ('byt5','lexical'):
            count=0
            for s in samples:
                ops=[dict(id=f'{i}:{j}',parents=[b['id']],texts=o['text'].split(),op='split')
                     for i,(b,options) in enumerate(zip(s['blocks'],s['candidates']))
                     for j,o in enumerate(options[1:],1) if origin in o['origins']]
                count+=structural_oracle(s['blocks'],ops,annotations(s))['correct']
            oracle[origin+'_only_correct']=count
        arms={};diagnostic_scores={};cer={}
        for name,predictions in pred[phase].items():
            if name not in ('baseline','historical_boundary') and f'seed{primary_seed}' not in name:continue
            distance=characters=0
            for s in samples:
                words=sorted(annotations(s),key=lambda w:((w['bbox'][1]+w['bbox'][3])/2,w['bbox'][0]))
                target=' '.join(w['text'].casefold() for w in words)
                text=' '.join(v.casefold() for v in predictions[s['sample_id']])
                distance+=Levenshtein.distance(text,target);characters+=len(target)
            cer[name]=dict(edit_distance=distance,care_gt_characters=characters,
                reading_order_cer=distance/characters if characters else None)
        for name,phase_changes in selected[phase].items():
            if f'seed{primary_seed}' not in name:continue
            stats=Counter();examples=[]
            for sid,changes in phase_changes.items():
                s=by[sid];index={b['id']:i for i,b in enumerate(s['blocks'])}
                for change in changes:
                    i=index[change['block_id']];label=labels[sid][i][change['candidate_index']-1]
                    category={0:'neutral',1:'beneficial',2:'damaging'}[label['label']]
                    stats[category]+=1;known=change['before'].casefold() in external
                    stats[category+'_known_external']+=int(known)
                    stats[category+'_with_context']+=int(any(s['contexts'][i]))
                    examples.append(dict(sample_id=sid,**change,local_label=category,
                        recovered_texts=label['recovered_texts'],damaged_texts=label['damaged_texts'],
                        known_external=known,left=[s['blocks'][j]['text'] for j in s['contexts'][i][0]],
                        right=[s['blocks'][j]['text'] for j in s['contexts'][i][1]]))
            arms[name]=dict(selected_edit_counts=dict(stats),examples=examples,
                aggregate=results['results'][phase][name]['aggregate'])
        for name,phase_scores in scores.items():
            if f'seed{primary_seed}' not in name:continue
            y=[];values=[];strata={True:([],[]),False:([],[])}
            for s in samples:
                sid=s['sample_id']
                for i,ls in enumerate(labels[sid]):
                    for label,v in zip(ls,phase_scores[phase][sid][i]):
                        beneficial=int(label['label']==1);y.append(beneficial);values.append(v)
                        sy,sv=strata[bool(any(s['contexts'][i]))];sy.append(beneficial);sv.append(v)
            def quality(ys,vs):
                return dict(candidates=len(ys),positive=sum(ys),
                    roc_auc=float(roc_auc_score(ys,vs)) if len(set(ys))>1 else None,
                    average_precision=float(average_precision_score(ys,vs)) if sum(ys) else None)
            diagnostic_scores[name]=dict(all=quality(y,values),
                with_context=quality(*strata[True]),without_context=quality(*strata[False]))
        out['cohorts'][phase]=dict(images=len(samples),pool_labels=dict(pool_stats),context=dict(context_stats),
            candidate_oracles=oracle,selected=arms,diagnostic_candidate_ranking=diagnostic_scores,
            reading_order_character_diagnostic=cer)
    # Known duplicate sensitivity: no retraining or threshold changes.
    subset=[s for s in data['test'] if s['sample_id'] not in c['known_duplicate_test_ids']]
    out['duplicate_excluded_test']={name:evaluate(subset,{s['sample_id']:p[s['sample_id']] for s in subset})['aggregate']
        for name,p in pred['test'].items() if name in ('baseline','historical_boundary') or f'seed{primary_seed}' in name}
    out['seed_stability']={phase:{arm:{str(seed):results['results'][phase][f'{arm}_seed{seed}_cap2']['aggregate']
        for seed in c['selector']['seeds']} for arm in ('lexical','word_lm','context','context_visual')}
        for phase in ('evaluation','test')}
    out['historical_boundary_comparison']={phase:bootstrap(results['results'][phase]['historical_boundary'],
        results['results'][phase][f'word_lm_seed{primary_seed}_cap2'],c['bootstrap_draws'],c['seed'])
        for phase in ('evaluation','test')}
    base=results['results']['test']['baseline']['aggregate']
    out['optimistic_fixed_pool_f1_bound']=dict(correct_count_ceiling=results['oracle']['test'],
        minimum_outputs=base['predictions'],care_words=base['gt_words'],
        bound=2*results['oracle']['test']/(base['predictions']+base['gt_words']),
        explanation='Loose upper bound: boundary-only choices cannot reduce original output count; not an exact F1-optimal oracle or a bound on future character corrections.')
    supplementary=root.parent/'threshold_sensitivity/results.json'
    if supplementary.exists():
        sensitivity=read(supplementary);choices=read(supplementary.parent/'selection_frozen.json')
        out['threshold_sensitivity']=dict(results_sha256=sha(supplementary),choices=choices['choices'],
            aggregates={phase:{name:v['aggregate'] for name,v in arms.items()} for phase,arms in sensitivity['results'].items()})
    write(root/'diagnostics.json',out)
    lines=['# Detailed contextual boundary analysis','','Approximate reproduction / post-hoc analysis; no policy changes.','',
           'Primary TEST includes300images/2204care words. All results are lexical, not spatial.','',
           '| Arm, primary seed | Correct | Outputs | Recovered | Damaged | F1 | CAL threshold cap2 |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in ['baseline','historical_boundary']+[f'{a}_seed{primary_seed}_cap2' for a in ('lexical','word_lm','context','context_visual')]:
        r=results['results']['test'][name]['aggregate']
        threshold=selection['calibration'][name.removesuffix('_cap2')]['2']['choice']['threshold'] if name.endswith('_cap2') else '—'
        lines.append(f"| {name} | {r['correct']} | {r['predictions']} | {r['recovered']} | {r['damaged']} | {r['f1']*100:.4f}% | {threshold} |")
    lines+=['','## Candidate availability versus selection','']
    for phase in ('calibration','evaluation','test'):
        d=out['cohorts'][phase];baseline=results['results'][phase]['baseline']['aggregate']['correct']
        lines.append(f"- {phase}: baseline{baseline}, joint candidate maximum{results['oracle'][phase]}, neural-only{d['candidate_oracles']['byt5_only_correct']}, lexical-only{d['candidate_oracles']['lexical_only_correct']}. These maximize correct count with GT; they are not achieved policies or F1 optima.")
        lines.append(f"  Local candidate labels: {d['pool_labels']}; context coverage: {d['context']}.")
    lines+=['','## Controlled effects and limits','']
    for arm in ('lexical','word_lm','context','context_visual'):
        key=f'{arm}_seed{primary_seed}_cap2';r=results['results']['test'][key]['aggregate'];b=results['comparisons']['test'][key]
        lines.append(f"- {arm}: delta F1{b['delta_f1']*100:+.4f}points, conditional95%CI{[round(v*100,4) for v in b['conditional_95_interval']]}; recovered{r['recovered']}/damaged{r['damaged']}. Selected edit labels: {out['cohorts']['test']['selected'][key]['selected_edit_counts']}.")
    lines+=['','Inspect matched word-LM→context and context→visual comparisons in results.json. Candidate ranking diagnostics are post-hoc discrimination measures, not new thresholds or evidence of causal context relevance.',
            '', 'Three selector seeds reuse fixed data and OOF proposals; identical outputs indicate algorithmic stability, not independently replicated training. Historical evaluation/TEST exploration and annotation conventions remain limitations.',
            '', 'The RGB arm uses proportional character positions and simple projection statistics. A negative result rejects this specific proxy; it does not reject visual boundary verification generally.',
            '', 'Read diagnostics.json for every accepted edit, beneficial/damaging/neutral examples, external-word/context strata and duplicate-excluded sensitivity. Neutral local edit counts need not sum to global per-image gains.']
    lines+=['','## Calibration sensitivity declared before primary results','',
        'The supplementary search uses every calibration decision breakpoint and preserves fixed models/candidates. It does not select thresholds on TEST.','',
        '| Arm | Objective, cap2 | Correct | Outputs | Recovered | Damaged | TEST F1 |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    if supplementary.exists():
        for name,value in sensitivity['results']['test'].items():
            if '_cap2_' not in name:continue
            r=value['aggregate'];lines.append(f"| {name.split('_seed')[0]} | {name.split('_cap2_')[1]} | {r['correct']} | {r['predictions']} | {r['recovered']} | {r['damaged']} | {r['f1']*100:.4f}% |")
    lines+=['',f"The fixed candidate pool has a loose optimistic F1 bound{out['optimistic_fixed_pool_f1_bound']['bound']*100:.4f}% because TP≤{results['oracle']['test']} and output count≥{base['predictions']}. This is limited to copied-character divisions in this experiment; it is not a ceiling for post-OCR correction generally."]
    (root/'analysis.md').write_text('\n'.join(lines)+'\n')
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path);args=parser.parse_args()
    result=analyze(args.run);print(json.dumps(dict(status='SUCCESS',results_sha256=result['results_sha256'])))


if __name__=='__main__':main()
