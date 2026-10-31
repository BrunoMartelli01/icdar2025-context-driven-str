"""Export audited ICDAR routing tables, cheap deployment heads and static plots."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time

import joblib
import numpy as np

from scripts.icdar_budget_data import read,sha,key
from recognition.icdar_budget_deployment import predict


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def report(root,bootstrap_draws):
    plan=read(root/'resolved_configuration.json');result=read(root/'results.json');frozen=read(root/'frozen_choices.json')
    audit=read(root/'supplemental_audit.json')
    assert audit['status']=='PASS' and audit['results_sha256']==sha(root/'results.json')
    records={r['sample_id']:r for data in plan['records'].values() for rows in data.values() for r in rows}
    output=root/'report';output.mkdir(exist_ok=True);rows=[];deployments={};qualification={};intervals={}
    for ds,arms in result['datasets'].items():
        deployments[ds]={};qualification[ds]={};intervals[ds]={}
        ids=plan['partitions'][ds]['test'];n=len(ids);rng=np.random.default_rng(plan['config']['seed'])
        weights=rng.multinomial(n,np.full(n,1/n),size=bootstrap_draws)
        for arm,values in arms.items():
            choice=frozen['choices'][ds][arm];bundle=joblib.load(root/choice['choice']['model_path'])
            data=read(root/'test_scores'/(ds+'__'+arm+'.json'));branches=data['branches'];policies=choice['policies']
            primary=next(p for p in values['thresholds'] if p['policy']['target_acceptance']==values['primary_acceptance'])
            batch=values['predeclared_batch_quotas'][str(values['primary_acceptance'])]
            points=[('all_light',None,values['all_light']),('all_deepsolo',None,values['all_deepsolo'])]
            points += [('validation_cut',p['policy']['target_acceptance'],p) for p in values['thresholds']]
            points += [('batch_budget',float(q),p) for q,p in values['predeclared_batch_quotas'].items()]
            points += [('conditional_gt_oracle',float(q),p) for q,p in values['conditional_gt_oracles'].items()]
            for kind,q,point in points:
                c=point['combined'];rows.append(dict(dataset=ds,expert_family=arm,expert=values['expert'],policy=kind,target_acceptance=q,
                    threshold=point['policy']['threshold'] if kind=='validation_cut' else None,f1_percent=100*c['f1'],accuracy_tn0_percent=100*c['accuracy_tn0'],
                    accepted_images=point['accepted_images'],deepsolo_images=point['deepsolo_images'],total_images=point['total_images'],deepsolo_percentage=point['deepsolo_percentage'],
                    tp=c['tp'],fp=c['fp'],fn=c['fn'],predicted_words=c['detected_words'],gt_words=c['gt_words'],ignored_outputs=c['ignored_outputs'],
                    result_class=plan['result_class'],routing_scope='logical cached routing'))
            # Fixed-rule bootstrap; no model or threshold reselection in resamples.
            base=np.array([[branches[s]['deepsolo'][k] for k in ('tp','detected_words','gt_words')] for s in ids]);base_draw=weights@base
            baseline=2*base_draw[:,0]/(base_draw[:,1]+base_draw[:,2]);intervals[ds][arm]={}
            for name,point in [('validation_cut',primary),('batch_budget',batch)]:
                keep=set(point['accepted_ids']);changed=np.array([[branches[s]['light' if s in keep else 'deepsolo'][k] for k in ('tp','detected_words','gt_words')] for s in ids])
                total=weights@changed;draws=2*total[:,0]/(total[:,1]+total[:,2])
                intervals[ds][arm][name]=dict(f1_interval95_percent=(100*np.quantile(draws,[.025,.975])).tolist(),
                    difference_from_all_deepsolo_interval95_pp=(100*np.quantile(draws-baseline,[.025,.975])).tolist(),
                    deepsolo_images=point['deepsolo_images'],total_images=n,deepsolo_percentage=point['deepsolo_percentage'],
                    scope='Image bootstrap conditional on fixed learned heads and fixed routing; excludes fit/selection uncertainty')
            bare_policies=[{k:p[k] for k in ['target_acceptance','threshold','tie_key','model_sha256']} for p in policies]
            # The older helper names this feature family "mask"; this study calls
            # it "maps". Correct only descriptive deployment metadata, not weights.
            bundle=dict(bundle,cost=dict(bundle['cost'],requires_unet_maps='maps' in bundle['groups']))
            deployment=dict(dataset=ds,expert=values['expert'],mode='approximate',claim_label='approximate reproduction',bundle=bundle,
                features=plan['config']['features'],policies=bare_policies,primary_acceptance=values['primary_acceptance'],
                source_model_sha256=choice['choice']['model_sha256'],plan_sha256=sha(root/'resolved_configuration.json'))
            dest=output/'deployments'/(ds+'__'+arm+'.joblib');dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():raise ValueError('Refuse to overwrite deployment')
            joblib.dump(deployment,dest)
            if choice['choice']['recipe']['kind']=='ridge':
                portable={k:v for k,v in deployment.items() if k!='bundle'}
                portable.update(groups=bundle['groups'],head=dict(mean=bundle['scaler'].mean_.tolist(),scale=bundle['scaler'].scale_.tolist(),
                    coef=bundle['regressor'].coef_.tolist(),intercept=float(bundle['regressor'].intercept_)))
                save(dest.with_suffix('.json'),portable)
            deployments[ds][arm]=dict(path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,
                source_head_bytes=(root/choice['choice']['model_path']).stat().st_size,feature_dimension=bundle['feature_dimension'],groups=bundle['groups'],
                recipe=choice['choice']['recipe'],additional_encoder_families=0,
                incremental_scope='Gate only; shared detector, OCR and selected expert are excluded')
            maximum_error=0.;seconds=[];decisions=0
            for sid in ids:
                r=records[sid];expert=read(root/'experts'/values['expert']/(key(sid)+'.json'));maps=None
                if 'maps' in bundle['groups']:
                    with np.load(expert['maps_path'],allow_pickle=False) as z:maps=z['probabilities']
                tick=time.perf_counter()
                single=predict(bundle,plan['config']['features'],expert['rows'],r['image_wh'],tie_key=r['rgb_sha256'],policies=bare_policies,probabilities=maps,frame=expert.get('frame'))
                seconds.append(time.perf_counter()-tick);expected=data['scores'][sid]
                assert (single['score'] is None)==(expected is None)
                if expected is not None:maximum_error=max(maximum_error,abs(expected-single['score']))
                for point in values['thresholds']:
                    assert single['accept_light'][str(point['policy']['target_acceptance'])]==(sid in point['accepted_ids'])
                    decisions+=1
            assert maximum_error<1e-9
            qualification[ds][arm]=dict(status='PASS',images=n,decisions=decisions,maximum_single_batch_score_error=maximum_error,
                cpu_gate_median_ms=1000*float(np.median(seconds)),cpu_gate_p95_ms=1000*float(np.quantile(seconds,.95)),
                scope='CPU single-image feature/head boundary with cached OCR/maps; excludes expert, disk I/O and RGB hashing; shared-host observational timing')
    fields=list(rows[0])
    with (output/'thresholds.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    save(output/'deployment_qualification.json',dict(status='PASS',datasets=qualification,deployments=deployments,code_sha256=sha(Path(__file__).parents[1]/'recognition/icdar_budget_deployment.py')))
    save(output/'intervals.json',dict(seed=plan['config']['seed'],bootstrap_draws=bootstrap_draws,datasets=intervals))
    lines=['# Dataset-specific trained routing — full TEST','', '**Approximate reproduction / research extension.** Lexical-multiset micro F1; casefold + alphanumeric normalization, no localization matching. Logical routing on cached outputs.','',
        '| Dataset | Expert | Rule | F1 % | Accepted light | DeepSolo images/total (%) |','| --- | --- | --- | ---: | ---: | --- |']
    for row in rows:
        primary=plan['config']['primary_acceptance'][row['dataset']]
        if row['policy'] in ['all_light','all_deepsolo'] or row['target_acceptance']==primary:
            lines.append(f"| {row['dataset']} | {row['expert_family']} ({row['expert']}) | {row['policy']} | {row['f1_percent']:.4f} | {row['accepted_images']} | {row['deepsolo_images']}/{row['total_images']} ({row['deepsolo_percentage']:.2f}%) |")
    lines += ['', 'Scalar thresholds are fixed on separate validation calibration images. Exact-budget ranks enforce the quota without TEST labels. Conditional GT oracles are descriptive upper bounds, never deployed. Bootstrap intervals exclude fitting/selection uncertainty. Paper Table5 accuracy and official polygon Hmean are different or unresolved protocols.', '',
        'See thresholds.csv for all predeclared controls; deployment_qualification.json for single-image replay and cheap head costs; supplemental_audit.json in the parent run for independent fitting/selection checks.']
    with (output/'SUMMARY.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    save(output/'receipt.json',dict(status='SUCCESS',results_sha256=sha(root/'results.json'),audit_sha256=sha(root/'supplemental_audit.json'),
        table_sha256=sha(output/'thresholds.csv'),summary_sha256=sha(output/'SUMMARY.md'),analysis_code_sha256=sha(__file__)))
    print(json.dumps(dict(status='SUCCESS',output=str(output),deployments=deployments)))


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=read(root/'results.json');fig,axes=plt.subplots(2,2,figsize=(17,12),sharey='col')
    colors={'yolo':'#1769aa','unet':'#d14900'}
    for column,(ds,arms) in enumerate(result['datasets'].items()):
        for row,field in enumerate(['predeclared_batch_quotas','thresholds']):
            ax=axes[row,column]
            for arm,values in arms.items():
                points=list(values[field].values()) if isinstance(values[field],dict) else values[field]
                points=sorted(points,key=lambda p:p['deepsolo_percentage']);color=colors[arm]
                ax.plot([p['deepsolo_percentage'] for p in points],[100*p['combined']['f1'] for p in points],marker='o',color=color,label=arm.upper())
                for index,p in enumerate(points):
                    offset=(10 if arm=='yolo' else -32)+index*2
                    if column==0 and row==1 and arm=='unet':offset=[-30,-60,-30][index]
                    ax.annotate(f"{p['deepsolo_images']}/{p['total_images']} ({p['deepsolo_percentage']:.2f}%)",(p['deepsolo_percentage'],100*p['combined']['f1']),textcoords='offset points',xytext=(0,offset),ha='center',fontsize=8,color=color)
            n=next(iter(arms.values()))['all_deepsolo']['total_images'];base=next(iter(arms.values()))['all_deepsolo']['combined']['f1']
            ax.axhline(100*base,color='gray',linestyle='--',label=f'DeepSolo only: {n}/{n} (100%)')
            ax.set_title(ds.upper()+(' — exact budget' if row==0 else ' — frozen validation cuts'));ax.set_xlabel('Images sent to DeepSolo (%)');ax.set_ylabel('Lexical micro F1 (%)');ax.grid(alpha=.2);ax.legend(fontsize=9);ax.margins(x=.25,y=.2)
    fig.suptitle('Approximate reproduction; cached routing; all official TEST images\nLabels: DeepSolo images/total (%) — no TEST threshold selection',fontsize=13);fig.tight_layout(rect=[0,0,1,.94])
    for ext in ('png','svg'):fig.savefig(root/'report'/('thresholds.'+ext),dpi=160)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--bootstrap-draws',type=int,default=2000);p.add_argument('--plot-only',action='store_true');args=p.parse_args()
    if args.plot_only:plot(args.run_root)
    else:report(args.run_root,args.bootstrap_draws)


if __name__=='__main__':main()
