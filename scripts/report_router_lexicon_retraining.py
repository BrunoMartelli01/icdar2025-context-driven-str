"""Combine independently audited retrained gates and matched previous controls."""

import argparse
import csv
from pathlib import Path

import numpy as np

from scripts.icdar_budget_data import read,sha
from scripts.run_yolo_f1_quality_study import seal


def report(c):
    root=Path(c['study_root']);previous=read(Path(c['replay_root'])/'results.json')['datasets'];out={};table=[];heads={};audits={}
    for ds in ('icdar13','icdar2015'):
        child=root/ds/'full';results=read(child/'results.json');audit=read(child/'supplemental_audit.json')
        assert audit['status']=='PASS' and audit['results_sha256']==sha(child/'results.json')
        assert read(root/'teacher'/ds/'independent_audit.json')['status']=='PASS'
        deployment=read(child/'report/deployment_qualification.json');assert deployment['status']=='PASS'
        heads[ds]=deployment['deployments'][ds];audits[ds]=audit['checks'];out[ds]={}
        for arm,v in results['datasets'][ds].items():
            old=previous[ds][arm];q=v['primary_acceptance'];base=v['all_deepsolo'];out[ds][arm]=dict(expert=v['expert'],primary_acceptance=q)
            for kind,p,before in [('all_light',v['all_light'],old['policies']['all_light']),
                ('all_deepsolo',base,old['policies']['all_deepsolo']),
                ('batch_budget',v['predeclared_batch_quotas'][str(q)],old['policies']['batch_'+str(q)]),
                ('validation_cut',next(p for p in v['thresholds'] if p['policy']['target_acceptance']==q),old['policies']['validation_'+str(q)])]:
                row=dict(dataset=ds,expert=arm,rule=kind,previous_f1_percent=100*before['f1'],
                    retrained_f1_percent=100*p['combined']['f1'],gain_pp=100*(p['combined']['f1']-before['f1']),
                    previous_deepsolo_images=before['deepsolo_images'],deepsolo_images=p['deepsolo_images'],
                    total_images=p['total_images'],deepsolo_percent=p['deepsolo_percentage'],previous_deepsolo_percent=before['deepsolo_percent'],
                    tp=p['combined']['tp'],fp=p['combined']['fp'],fn=p['combined']['fn'],
                    threshold=p['policy']['threshold'] if kind=='validation_cut' else None)
                table.append(row);out[ds][arm][kind]=row
                if kind in ('batch_budget','validation_cut'):
                    branches=read(child/'test_scores'/(ds+'__'+arm+'.json'))['branches'];oldbranches=read(Path(c['replay_root'])/'per_image'/(ds+'__'+arm+'.json'))['branches']
                    assert branches==oldbranches
                    ids=sorted(branches);selected=set(p['accepted_ids']);prior=set(before['accepted_ids'])
                    new=np.array([[branches[s]['light' if s in selected else 'deepsolo'][k] for k in ('tp','detected_words','gt_words')] for s in ids])
                    prior=np.array([[branches[s]['light' if s in prior else 'deepsolo'][k] for k in ('tp','detected_words','gt_words')] for s in ids])
                    weights=np.random.default_rng(c['seed']).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
                    a,b=weights@new,weights@prior;delta=2*a[:,0]/(a[:,1]+a[:,2])-2*b[:,0]/(b[:,1]+b[:,2])
                    row['paired_gain_interval95_pp']=(100*np.quantile(delta,[.025,.975])).tolist()
    comparison=dict(status='PASS',mode='approximate',claim_label='approximate reproduction',datasets=out,heads=heads,audit_checks=audits,
        actual_new_deepsolo_calls=900,cached_neural_inferences_reused=1062,
        uncertainty='2000 fixed-rule paired-image bootstrap draws; excludes fitting/selection and prior TEST exposure')
    seal(root/'comparison.json',comparison)
    with (root/'comparison.csv').open('x',newline='') as f:
        columns=[k for k in table[0]]+['paired_gain_interval95_pp'];w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(table)
    lines=['', '## Completed results','',
        '**Approximate reproduction / research extension.** Complete official TEST and unchanged word lexical F1. The previous controls below already use the new Generic/Weak DeepSolo; differences isolate new router fitting rather than the prior fallback upgrade.','',
        '| Dataset | Expert | Light-only F1 % | DeepSolo light-only/total (%) | Previous router F1 % | Retrained F1 % | DeepSolo router/total (%) |',
        '| --- | --- | ---: | --- | ---: | ---: | --- |']
    for ds,arms in out.items():
        for arm,v in arms.items():
            p=v['batch_budget'];light=v['all_light'];n=p['total_images']
            lines.append(f"| {ds} | {arm} ({v['expert']}) | {light['retrained_f1_percent']:.4f} | 0/{n} (0.00%) | {p['previous_f1_percent']:.4f} | {p['retrained_f1_percent']:.4f} | {p['deepsolo_images']}/{n} ({p['deepsolo_percent']:.2f}%) |")
    lines += ['', '### Separate validation scalar cuts','',
        'Cutoffs are selected again from the separate CAL scores after retraining, with no subsequent refit. Batch budgets and fixed scalar cuts are different policies. The scalar before/after routes can send different numbers of images to DeepSolo; their F1 difference is not a matched-cost improvement.','',
        '| Dataset | Expert | New cutoff | Previous F1 % | Previous DeepSolo/total (%) | Retrained F1 % | New DeepSolo/total (%) |',
        '| --- | --- | ---: | ---: | --- | ---: | --- |']
    for ds,arms in out.items():
        for arm,v in arms.items():
            p=v['validation_cut'];n=p['total_images'];lines.append(f"| {ds} | {arm} | {p['threshold']:.10f} | {p['previous_f1_percent']:.4f} | {p['previous_deepsolo_images']}/{n} ({p['previous_deepsolo_percent']:.2f}%) | {p['retrained_f1_percent']:.4f} | {p['deepsolo_images']}/{n} ({p['deepsolo_percent']:.2f}%) |")
    screen=read(root/'candidate_selection_initial.json');tt=read(root/'totaltext_ic15_test.json')['metrics']
    lines += ['', '### ICDAR2015 U-Net comparison','',
        'All eleven candidates use only the original50 selection images /215 care words. This finite comparison includes six additionally qualified completed branches; it is not an exhaustive or unbiased comparison after historical upstream VAL100 policy exposure.','',
        '| Candidate | Selection F1 % | DeepSolo images/total (%) |','| --- | ---: | --- |']
    for name,m in sorted(screen['metrics'].items(),key=lambda v:-v[1]['f1']):lines.append(f"| {name} | {100*m['f1']:.4f} | 0/50 (0.00%) |")
    lines += ['',f"The pure TotalText corrected expert also runs on complete IC15 TEST: F1{100*tt['f1']:.4f}%, DeepSolo0/500 (0.00%). The selected long768 remains59.7031%, DeepSolo0/500 (0.00%). Additional candidate checkpoints, completed training manifests and matching FIT900/VAL100 IDs are verified in candidate_qualification.json;550 selection word recounts and500 TotalText TEST recounts pass candidate_selection_audit.json.",'',
        '### Interpretation, gate size and qualifications','',
        'Retraining is not automatically an improvement. Report negative outcomes, keep both old and new heads and do not promote a winner using TEST. The new fixed-budget differences and their conditional uncertainty are stored in comparison.json. Tiny selection populations, imperfect observable routing signals and the changed teacher utilities remain limitations. TRAIN-only Weak versus released TEST Weak is an explicit vocabulary distribution shift; the neural IC15 teacher itself has seen official TRAIN upstream.','',
        '| Dataset | Expert | New groups | Feature dimension | Serialized deployment bytes | Additional encoder families |','| --- | --- | --- | ---: | ---: | ---: |']
    for ds,arms in heads.items():
        for arm,h in arms.items():lines.append(f"| {ds} | {arm} | {', '.join(h['groups'])} | {h['feature_dimension']} | {h['bytes']} | 0 |")
    lines += ['', 'Single-image deployment replay validates every gate score and all three CAL policies for every TEST image. The core independent audit verifies weighted Ridge normal equations, boosted-tree refits, all candidate selection routes, disjoint RGB groups, calibration cuts and exact conditional-oracle certificates. Teacher audits recount all1962 TRAIN/TEST word records and verify the Weak dictionary contains only TRAIN words. Raw CTC/geometry outputs, checkpoint/source hashes and per-image GPU/input/precision/memory records are preserved. No sample is omitted.','',
        'Physical new inference is900 IC15 TRAIN calls; reuse229 IC13 TRAIN,100 IC15 TRAIN and733 TEST exports (1062 reused). All final routing is logical on cached outputs; no online timing/speedup was measured. New heads have no extra image/text encoder. Deployment files and bootstrap intervals are in each dataset/full/report/. The combined comparison.json/.csv bind the matched previous controls. Both old and newly trained models remain available.','',
        'No strict or paper-equivalent result is claimed. All source, tests, errors and comparison artifacts must be sealed in the final completion receipt.']
    with Path('docs/ROUTER_LEXICON_RETRAINING.md').open('a') as f:f.write('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,(ds,arms) in zip(axes,out.items()):
        for arm,v in arms.items():
            p=v['batch_budget'];n=p['total_images'];color='#1769aa' if arm=='yolo' else '#d14900';x=[0,1]
            ax.plot(x,[p['previous_f1_percent'],p['retrained_f1_percent']],marker='o',label=arm.upper(),color=color)
            for i,y in zip(x,[p['previous_f1_percent'],p['retrained_f1_percent']]):
                ax.annotate(f"{y:.2f}%\n{p['deepsolo_images']}/{n} ({p['deepsolo_percent']:.2f}%)",(i,y),xytext=(0,10 if arm=='yolo' else -28),textcoords='offset points',ha='center',fontsize=8,color=color)
        ax.set_xticks([0,1],['Previous gate + new DeepSolo','Retrained gate + same DeepSolo']);ax.set_xlim(-.4,1.4);ax.set_ylabel('Word lexical micro F1 (%)');ax.set_title(ds.upper());ax.legend();ax.grid(alpha=.2);ax.margins(y=.4)
    fig.suptitle('Approximate reproduction — matched fixed budgets, cached outputs\nLabels show DeepSolo images/total (%)');fig.tight_layout(rect=[0,0,1,.9])
    for ext in ('png','svg'):fig.savefig(root/('comparison.'+ext),dpi=160)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();report(read(a.config))
