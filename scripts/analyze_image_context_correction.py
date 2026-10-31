"""Post-hoc failure analysis of frozen image-context correction experiments."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from recognition.image_context_correction import apply_choices, select_subset
from recognition.text_reconstruction import structural_oracle
from scripts.run_image_context_correction import operation_label
from scripts.run_contextual_boundary_selection import evaluate
from scripts.run_text_reconstruction_tests import annotations,read,sha,write


def analyze(root):
    root=Path(root);c=read(root/'resolved_config.json');data=read(root/'pool.json');result=read(root/'results.json')
    scores=read(root/'scores.json');selection=read(root/'selection_frozen.json');pred=read(root/'predictions.json')
    output=dict(claim_label='approximate reproduction / post-hoc exploratory diagnosis',results_sha256=sha(root/'results.json'),
        analysis_code_sha256=sha(Path(__file__)),cohorts={})
    for phase in ('calibration','evaluation','test'):
        label_data={};pool_counts=Counter();oracles={};rank={};examples={};memory=[]
        for s in data[phase]:
            words=annotations(s);labels=[operation_label(s,o,words,c) for o in s['pool']];label_data[s['sample_id']]=labels
            pool_counts.update((o['kind'],label['label']) for o,label in zip(s['pool'],labels))
        for subset in ('boundary','character','fragment','expanded'):
            rows=[]
            for s in data[phase]:
                ops=[dict(id='candidate:'+str(i),parents=[s['blocks'][j]['id'] for j in s['pool'][i]['parents']],texts=s['pool'][i]['text'].split())
                     for i in select_subset(s['pool'],subset)]
                rows.append(dict(sample_id=s['sample_id'],**structural_oracle(s['blocks'],ops,annotations(s))))
            correct=sum(r['correct'] for r in rows);g=result['results'][phase]['baseline']['aggregate']['gt_words']
            oracles[subset]=dict(maximum_correct=correct,perfect_precision_f1_upper_bound=2*correct/(correct+g),per_image=rows,
                caveat='GT-assisted maximum correct count, not deployable policy or F1 optimum; perfect-precision bound assumes free FP deletion.')
        for name,record in selection['models'].items():
            ys=[];vs=[];stats=Counter();cases=[];threshold=record['choices']['correct_cap2']['threshold']
            for s in data[phase]:
                sid=s['sample_id'];indices=select_subset(s['pool'],record['spec']['subset']);values=scores[name][phase][sid];groups=defaultdict(list)
                for index,value in zip(indices,values):
                    label=label_data[sid][index];ys.append(int(label['label']==1));vs.append(value)
                    groups[tuple(s['pool'][index]['parents'])].append((index,value,label))
                _,chosen=apply_choices(s['blocks'],[s['pool'][i] for i in indices],values,threshold)
                accepted={indices[j] for j in chosen}
                for parents,options in groups.items():
                    useful=any(label['label']==1 for _,_,label in options)
                    best=max(options,key=lambda row:(row[1],-row[0]))
                    if useful:
                        stats['groups_with_beneficial_alternative']+=1
                        stats['top_beneficial' if best[2]['label']==1 else 'top_not_beneficial']+=1
                        if best[2]['label']==1:stats['accepted_beneficial_top' if best[0] in accepted else 'rejected_beneficial_top']+=1
                    for index,value,label in options:
                        if index in accepted:
                            op=s['pool'][index];stats['accepted_'+str(label['label'])]+=1
                            if len(cases)<30:cases.append(dict(sample_id=sid,before=[s['blocks'][j]['text'] for j in parents],after=op['text'],kind=op['kind'],score=value,**label))
            rank[name]=dict(alternatives=len(ys),beneficial=sum(ys),average_precision=float(average_precision_score(ys,vs)) if sum(ys) else None,**dict(stats))
            examples[name]=cases
        output['cohorts'][phase]=dict(pool_label_counts={f'{kind}:{label}':count for (kind,label),count in pool_counts.items()},oracles=oracles,ranking=rank,accepted_examples=examples)
    subset=[s for s in data['test'] if s['sample_id'] not in c['known_duplicate_test_ids']]
    output['duplicate_excluded_test']={name:evaluate(subset,{s['sample_id']:arm[s['sample_id']] for s in subset})['aggregate'] for name,arm in pred['test'].items()}
    output['generator_receipts']={path.parent.name:read(path) for path in sorted((root/'generators').glob('*/receipt.json'))}
    training={}
    for path in sorted((root/'generators').glob('*/pairs.json')):
        rows=read(path);counts=Counter()
        for row in rows['records']:
            source=' '.join(row['source'].replace('|',' ').split()).casefold();target=' '.join(row['target'].split()).casefold()
            category='identity' if source==target else 'boundary' if source.replace(' ','')==target.replace(' ','') else 'characters'
            counts[(row['kind'],category)]+=1
        training[path.parent.name]=dict(kinds_and_edits={f'{kind}:{edit}':count for (kind,edit),count in counts.items()},
            unique_sources=len({r['source'].casefold() for r in rows['records']}),unique_targets=len({r['target'].casefold() for r in rows['records']}),
            conflicting_sources_removed=rows['conflicting_sources_removed'])
    output['training_distribution']=training
    write(root/'diagnostics.json',output)
    lines=['# Detailed image-context correction analysis','','Approximate reproduction on historically explored transcription cohorts.','',
           'All thresholds were selected on calibration before this run evaluated TEST. Results do not establish independent generalization.','',
           '| TEST comparison | F1 change, percentage points | Conditional 95% interval |','| --- | ---: | --- |']
    for key,row in result['comparisons']['test'].items():
        if '_versus_' in key:lines.append(f"| {key} | {row['delta_f1']*100:+.4f} | {[round(v*100,4) for v in row['conditional_95_interval']]} |")
    lines+=['','## Candidate availability and decision failures','']
    for phase in ('evaluation','test'):
        d=output['cohorts'][phase];base=result['results'][phase]['baseline']['aggregate']['correct']
        lines.append(f"- {phase}: baseline {base}; maximum correct-count pool oracles "+', '.join(f"{k}={v['maximum_correct']}" for k,v in d['oracles'].items())+'.')
        for name in ('boundary_word','boundary_ordered','boundary_inventory','boundary_global_context','boundary_wrong_context','boundary_global_listwise','expanded_global'):
            r=d['ranking'][name];actual=result['results'][phase][name+'_correct_cap2']['aggregate']
            lines.append(f"  {name}: {actual['recovered']} recovered/{actual['damaged']} damaged; AP={r['average_precision']}; useful groups={r.get('groups_with_beneficial_alternative',0)}, beneficial tops accepted/rejected={r.get('accepted_beneficial_top',0)}/{r.get('rejected_beneficial_top',0)}.")
    lines+=['','## Interpretation limits','',
        'Compare correct-image context with both target-only and wrong-image controls on the same pool. A higher descriptive TEST endpoint alone does not prove a context effect; consult paired intervals and evaluation transfer.',
        '', 'Raw generator recipes share architecture, optimizer and update budget. Clean alignment, record mix and noise differ; this is a whole-recipe comparison, not proof that one noise operation alone causes a gain.',
        '', 'FP filters may improve F1 by removing actual text with incorrect transcription. Presence labels measure care-GT coverage, not physical text presence. Damage caps constrain calibration only.',
        '', 'Pool oracles use GT and maximize correct count. Their perfect-precision bounds assume ideal deletion of all remaining unmatched outputs, so they are not deployable or F1-optimal methods.',
        '', 'Read diagnostics.json for all cohort counts, selected-edit labels, examples, duplicate-excluded results and actual per-stage memory. No TEST winner is promoted to defaults.']
    lines+=['','Expanded-pool rejection of a beneficial top proposal can also reflect overlap with another selected operation; it is not solely a threshold failure.',
        '', 'OOF models train on fewer FIT images than the final models despite matched record/update budgets. This distribution difference remains a limitation.',
        '', 'Boundary-pool membership in this battery is defined by equality of compact casefold strings. The historical strict Unicode character-copy control remains a separate reference.']
    (root/'analysis.md').write_text('\n'.join(lines)+'\n');return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path);args=parser.parse_args();out=analyze(args.run)
    print(json.dumps(dict(status='SUCCESS',results_sha256=out['results_sha256'])))


if __name__=='__main__':main()
