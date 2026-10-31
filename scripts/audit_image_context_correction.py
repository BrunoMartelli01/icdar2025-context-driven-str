"""Independent count, frozen-policy, context-coverage and OOF provenance audit."""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import pickle

import numpy as np

from recognition.image_context_correction import apply_choices, select_subset
from scripts.run_text_reconstruction_tests import annotations, read, sha, write


def independent_count(texts,words):
    pred=Counter(t.casefold() for text in texts for t in (text.split() or ['']))
    gt=Counter(w['text'].casefold() for w in words)
    match=pred & gt
    return dict(correct=sum(match.values()),predictions=sum(pred.values()),gt_words=sum(gt.values())),match


def audit(root):
    from transformers import AutoTokenizer
    from scripts.run_image_context_correction import predict_selector,raw_generator_predictions,filter_predictions
    root=Path(root);c=read(root/'resolved_config.json');protocol=read(root/'protocol.json');result=read(root/'results.json')
    data=read(root/'pool.json');predictions=read(root/'predictions.json');selection=read(root/'selection_frozen.json')
    scores=read(root/'scores.json');filters=read(root/'filter_selection_frozen.json');fscores=read(root/'filter_scores.json')
    counts=Counter()
    assert not selection['evaluation_test_used_for_selection'] and not filters['evaluation_test_used_for_selection']
    assert sha(root/'selection_frozen.json')==result['selection_sha256']
    assert sha(root/'filter_selection_frozen.json')==result['filter_selection_sha256']
    assert sha(root/'scores.json')==selection['scores_sha256']
    assert sha(root/'filter_scores.json')==filters['scores_sha256']
    for path,expected in c['input_sha256'].items():assert sha(path)==expected,path
    for path,expected in protocol['code_sha256'].items():assert sha(Path(__file__).resolve().parents[1]/path)==expected,path
    folds=protocol['folds'];fit={s['sample_id'] for s in data['fit']}
    for recipe in c['generator']['recipes']:
        seen=set()
        for fold in [*range(c['folds']),'final']:
            work=root/'generators'/f'{recipe}_{fold}';pairs=read(work/'pairs.json');receipt=read(work/'receipt.json')
            expected=fit if fold=='final' else {sid for sid in fit if folds[sid]!=fold}
            assert set(pairs['training_ids'])==expected
            if fold!='final':
                held={sid for sid in fit if folds[sid]==fold}
                assert set(pairs['prediction_ids'])==held and not expected&held
                seen|=held
            for row in pairs['records']:
                if 'source_ids' in row:assert set(row['source_ids'])<=expected
                else:assert recipe=='historical' and row['source_ids_scope']=='fold_training_ids'
                assert len(('repair: '+row['source']).encode())+1<=c['generator']['max_input_bytes']
                assert len(row['target'].encode())+1<=c['generator']['max_target_bytes']
                counts['training_pairs']+=1
            if not protocol['smoke']:assert len(pairs['records'])==c['generator']['training_pairs']
            assert receipt['updates']==int(np.ceil(len(pairs['records'])/c['generator']['batch_size']))
            assert sha(work/'checkpoint.pt')==receipt['checkpoint_sha256']
            assert sha(work/'predictions.json')==receipt['predictions_sha256']
            assert not receipt['oom'] and all(not receipt['loading'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
            counts['generator_stages']+=1
        assert seen==fit
    tokenizer=AutoTokenizer.from_pretrained(c['paths']['bert_snapshot'],local_files_only=True,use_fast=True)
    for phase,samples in data.items():
        by={s['sample_id']:s for s in samples};evidence=read(root/f'language_{phase}.json')
        for s in samples:
            for record in evidence[s['sample_id']]['context_audit']:
                source=by[record['source_sample_id']]
                excluded=set(record['parents']) if record['mode']!='wrong' else set()
                expected=[t for i,b in enumerate(source['blocks']) if i not in excluded and b['text'].strip()
                          for t in tokenizer.encode(b['text'],add_special_tokens=False)]
                actual=[t for w in record['windows'] for t in w['token_ids']]
                assert expected==actual,(s['sample_id'],record['mode'])
                assert all(len(w['token_ids'])<=record['window_budget'] for w in record['windows'])
                if record['mode']!='wrong':assert all(not excluded&set(w['source_indices']) for w in record['windows'])
                elif len(samples)>1:assert record['source_sample_id']!=s['sample_id']
                counts['context_windows']+=len(record['windows']);counts['context_groups']+=1
            for op in s['pool']:
                assert op['parents'] and len(set(op['parents']))==len(op['parents'])
                assert op['text'].split()
                if op['kind']=='boundary':assert ''.join(s['blocks'][i]['text'] for i in op['parents']).casefold()==''.join(op['text'].split()).casefold()
                counts['candidate_operations']+=1
    with (root/'selectors.pkl').open('rb') as stream:models=pickle.load(stream)
    for phase in ('calibration','evaluation','test'):
        samples=data[phase];features=read(root/f'features_{phase}.json')
        for name,record in selection['models'].items():
            spec=record['spec']
            for s in samples:
                indices=select_subset(s['pool'],spec['subset'])
                replay=predict_selector(models[name],[features[s['sample_id']]['operations'][i][spec['feature']] for i in indices])
                assert np.allclose(replay,scores[name][phase][s['sample_id']],rtol=0,atol=1e-7)
                for key,choice in [*record['choices'].items(),('raw',dict(threshold=0.))]:
                    output,_=apply_choices(s['blocks'],[s['pool'][i] for i in indices],replay,choice['threshold'])
                    assert output==predictions[phase][name+'_'+key][s['sample_id']]
                    counts['policy_replays']+=1
        for recipe in c['generator']['recipes']:
            assert raw_generator_predictions(root,c,samples,phase,recipe)==predictions[phase]['generator_'+recipe+'_raw']
            counts['generator_policy_replays']+=len(samples)
        for policy,record in filters['selection'].items():
            for key,choice in record['choices'].items():
                assert filter_predictions(samples,fscores[phase],choice['threshold'],policy,choice['presence_threshold'])==predictions[phase]['filter_'+policy+'_'+key]
                counts['filter_policy_replays']+=len(samples)
        for name,arm in predictions[phase].items():
            totals=Counter();expected_rows=result['results'][phase][name]['per_image']
            assert len(expected_rows)==len(samples)
            for s,record in zip(samples,expected_rows):
                words=annotations(s);count,matched=independent_count(arm[s['sample_id']],words)
                _,before=independent_count([b['text'] for b in s['blocks']],words)
                count.update(recovered=sum((matched-before).values()),damaged=sum((before-matched).values()))
                assert record['sample_id']==s['sample_id']
                assert all(record[k]==v for k,v in count.items()),(phase,name,s['sample_id'])
                totals.update(count);counts['image_arm_counts']+=1
            total=result['results'][phase][name]['aggregate']
            assert all(total[k]==v for k,v in totals.items())
            assert np.isclose(total['f1'],2*totals['correct']/(totals['predictions']+totals['gt_words']))
    for record in selection['models'].values():
        for key,choice in record['choices'].items():
            cap=int(key.rsplit('cap',1)[1]);valid=[r for r in record['curve'] if r['damaged']<=cap]
            objective=key.split('_')[0]
            best=max(valid,key=lambda r:(r['correct'],r['f1'],r['threshold']) if objective=='correct' else (r['f1'],r['correct'],r['threshold']))
            assert choice==best;counts['calibration_choices']+=1
    output=dict(status='PASS',results_sha256=sha(root/'results.json'),audit_code_sha256=sha(Path(__file__)),checks=dict(counts),
        limitation='Code-level leakage boundaries require tests; this audit validates saved provenance and outputs, not historical independence.')
    write(root/'independent_audit.json',output);return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path);args=parser.parse_args()
    print(json.dumps(audit(args.run)))


if __name__=='__main__':main()
