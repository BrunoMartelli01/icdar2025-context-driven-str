"""Predeclared CAL-only composition of frozen image-context correction and FP filtering."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path

from recognition.image_context_correction import select_subset
from recognition.image_context_joint_policy import compose
from scripts.audit_image_context_correction import independent_count
from scripts.run_contextual_boundary_selection import bootstrap
from scripts.run_text_reconstruction_tests import annotations,read,sha,write

REPO=Path(__file__).resolve().parents[1]


def score(samples,pred,truth):
    rows=[]
    for s in samples:
        sid=s['sample_id'];counts,matched=independent_count(pred[sid],truth[sid])
        _,before=independent_count([b['text'] for b in s['blocks']],truth[sid])
        counts.update(recovered=sum((matched-before).values()),damaged=sum((before-matched).values()))
        rows.append(dict(sample_id=sid,**counts))
    total={k:sum(r[k] for r in rows) for k in ('correct','predictions','gt_words','recovered','damaged')}
    t,n,g=(total[k] for k in ('correct','predictions','gt_words'))
    total.update(fp=n-t,fn=g-t,precision=t/n if n else 0.,recall=t/g if g else 0.,f1=2*t/(n+g) if n+g else 0.)
    return dict(aggregate=total,per_image=rows)


def predict(samples,pool_scores,filter_scores,subset,policy,order,choice):
    output={};traces={}
    for s in samples:
        sid=s['sample_id'];indices=select_subset(s['pool'],subset);f=filter_scores[sid]
        output[sid],traces[sid]=compose(s['blocks'],[s['pool'][i] for i in indices],pool_scores[sid],choice['correction_threshold'],
            f['exact'],f['presence'],choice['filter_threshold'],choice['presence_threshold'],policy,order)
    return output,traces


def run(config,output,mode):
    config=Path(config).resolve();output=Path(output)
    if mode!='approximate' or REPO/'configs/approximate' not in config.parents:raise ValueError('Explicit APPROXIMATE config required')
    c=read(config)
    if c['reproduction_mode']!=mode or c['experiment_kind']!='image_context_joint_policy_v1':raise ValueError('This is an approximate supplementary experiment')
    if output.exists():raise ValueError('Use a fresh joint-policy output')
    root=Path(c['source_run']);base=read(root/'resolved_config.json')
    for path,expected in c['input_sha256'].items():
        if sha(path)!=expected:raise ValueError('Joint-policy scientific input changed: '+path)
    if not (root/'completion.json').exists():raise ValueError('Primary run must complete successfully')
    assert read(root/'completion.json')['results_sha256']==sha(root/'results.json')
    data=read(root/'pool.json');scores=read(root/'scores.json');fscores=read(root/'filter_scores.json')
    source_selection=read(root/'selection_frozen.json');selection={};output.mkdir(parents=True)
    write(output/'resolved_config.json',c)
    write(output/'protocol.json',dict(claim_label='approximate reproduction',source_results_sha256=sha(root/'results.json'),
        declared_before_primary_results=c['declared_before_primary_results'],
        code_sha256={p:sha(REPO/p) for p in ['recognition/image_context_joint_policy.py','scripts/run_image_context_joint_policy.py']}))
    # Only CAL annotations are read before the complete supplementary policy file is frozen.
    truth={s['sample_id']:annotations(s) for s in data['calibration']}
    for model,policy,order in itertools.product(c['correction_models'],c['filter_policies'],c['orders']):
        name=model+'__'+policy+'__'+order;subset=source_selection['models'][model]['spec']['subset'];curve=[]
        for ct,ft,pt in itertools.product(base['selector']['thresholds'],base['filter']['thresholds'],base['filter']['presence_thresholds'] if policy=='presence' else [0.]):
            choice=dict(correction_threshold=ct,filter_threshold=ft,presence_threshold=pt)
            pred,_=predict(data['calibration'],scores[model]['calibration'],fscores['calibration'],subset,policy,order,choice)
            curve.append(dict(**choice,**score(data['calibration'],pred,truth)['aggregate']))
        choices={}
        for cap,objective in itertools.product(c['damage_limits'],c['objectives']):
            valid=[r for r in curve if r['damaged']<=cap]
            key=lambda r:((r['correct'],r['f1'],r['correction_threshold'],-r['filter_threshold'],-r['presence_threshold']) if objective=='correct' else
                          (r['f1'],r['correct'],r['correction_threshold'],-r['filter_threshold'],-r['presence_threshold']))
            choices[f'{objective}_cap{cap}']=max(valid,key=key)
        selection[name]=dict(model=model,subset=subset,policy=policy,order=order,curve=curve,choices=choices)
        print('JOINT_CALIBRATED',name,json.dumps(choices['f1_cap2']),flush=True)
    write(output/'selection_frozen.json',dict(policies=selection,evaluation_test_used_for_selection=False))
    results={};predictions={};comparisons={};source=read(root/'results.json')
    for phase in ('calibration','evaluation','test'):
        truth={s['sample_id']:annotations(s) for s in data[phase]};arms={}
        for name,record in selection.items():
            for key,choice in record['choices'].items():
                pred,_=predict(data[phase],scores[record['model']][phase],fscores[phase],record['subset'],record['policy'],record['order'],choice)
                arms[name+'__'+key]=pred
        predictions[phase]=arms;results[phase]={name:score(data[phase],p,truth) for name,p in arms.items()}
        if phase!='calibration':
            comparisons[phase]={name:bootstrap(source['results'][phase]['baseline'],r,base['bootstrap_draws'],base['seed']) for name,r in results[phase].items()}
    write(output/'predictions.json',predictions)
    write(output/'results.json',dict(status='SUCCESS',claim_label='approximate reproduction',selection_sha256=sha(output/'selection_frozen.json'),results=results,comparisons=comparisons,
        routing={p:dict(images_sent_to_deepsolo=0,total_evaluated_images=len(data[p]),percentage=0.,execution='offline post-OCR composition; no online DeepSolo inference') for p in ('calibration','evaluation','test')},
        limitations=['Predeclared supplementary composition on historically explored cohorts, not independent validation.',
            'Larger CAL policy search can overfit the100-image calibration cohort.','Correct-first protects selected edits; it does not certify their correctness.']))
    # Independent common evaluator rechecks every per-image result and denominator after freezing.
    from scripts.run_contextual_boundary_selection import evaluate
    checked=0
    for phase,arms in predictions.items():
        for name,p in arms.items():
            expected=evaluate(data[phase],p)['aggregate'];actual=results[phase][name]['aggregate']
            assert expected==actual,(phase,name,expected,actual);checked+=len(data[phase])
    write(output/'independent_audit.json',dict(status='PASS',image_policy_checks=checked,results_sha256=sha(output/'results.json')))
    lines=['# Joint correction and FP-filter composition','','Approximate reproduction; CAL-only predeclared supplementary policies.','',
        '| TEST policy, F1-first cap2 | Correct | Outputs | FP | Recovered | Damaged | F1 | DeepSolo images |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for name,value in results['test'].items():
        if not name.endswith('f1_cap2'):continue
        r=value['aggregate'];lines.append(f"| {name} | {r['correct']} | {r['predictions']} | {r['fp']} | {r['recovered']} | {r['damaged']} | {r['f1']*100:.4f}% | 0/{len(data['test'])} (0.00%) |")
    (output/'analysis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='SUCCESS',image_policy_checks=checked)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',required=True);parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();run(args.config,args.output,args.mode)


if __name__=='__main__':main()
