"""Descriptive score diagnostics and publication-style plots of an audited study.

No policy is selected or changed using TEST labels. All TEST rankings below
are descriptive measurements of the already declared methods and rank budgets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from recognition.f1_quality_study import evaluate, good, sorted_ids


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discrimination(scores, labels):
    """Pairwise AUROC and threshold-block AP, including missing scores at bottom."""
    pairs=sorted(zip(scores,labels),key=lambda x:float('-inf')if x[0]is None else x[0],reverse=True)
    positives=sum(labels);negatives=len(labels)-positives
    if not positives or not negatives:return {'auroc':None,'average_precision':None}
    tp=fp=0;area=ap=0.;i=0
    while i<len(pairs):
        j=i+1
        while j<len(pairs)and pairs[j][0]==pairs[i][0]:j+=1
        pos=sum(y for _,y in pairs[i:j]);neg=j-i-pos
        # Higher-scored positives outrank this block's negatives; ties count half.
        area+=tp*neg+pos*neg/2
        tp+=pos;fp+=neg;ap+=pos/positives*tp/(tp+fp);i=j
    return {'auroc':area/(positives*negatives),'average_precision':ap,
        'positives':positives,'negatives':negatives,'missing_scores':sum(s is None for s in scores)}


def paired_intervals(branches,ids,accepted,reference,weights):
    def resampled(a):
        rows=np.array([[branches[s]['light'if s in a else'deepsolo'][k]
            for k in ('tp','fp','fn')]for s in ids])
        counts=weights@rows
        return 2*counts[:,0]/(2*counts[:,0]+counts[:,1]+counts[:,2])
    difference=resampled(set(accepted))-resampled(set(reference))
    return np.quantile(difference,[.025,.975]).tolist()


def analyze(root):
    plan=read(root/'resolved_configuration.json');result=read(root/'results.json')
    audit=read(root/'independent_audit.json');assert audit['status']=='PASS'
    assert audit['results_sha256']==sha(root/'results.json')
    cfg=plan['config'];source=Path(plan['paths']['sources']['coverage'])
    branches=read(source/'results.json')['resolutions']['640']['branch_metrics']
    features={split:{v['sample_id']:v['scores']for p in(root/split/'features').glob('*.json')
        for v in [read(p)]}for split in ('train','test')}
    out=root/'analysis';out.mkdir(exist_ok=True)
    diagnostics={'reproduction_mode':'approximate','claim_label':'approximate reproduction',
        'analysis_class':'POSTHOC_DESCRIPTIVE_NO_NEW_DEPLOYMENT_SELECTION',
        'results_sha256':sha(root/'results.json'),'audit_sha256':sha(root/'independent_audit.json'),
        'analysis_code_sha256':sha(__file__),'no_training':True,'no_new_model_inference':True,
        'gates_unchanged':True,'splits':{},'paired_intervals':{}}
    rows=[]
    populations=[('train','train',sorted(features['train'])),('selection','train',plan['selection_ids']),
        ('calibration','train',plan['threshold_calibration_ids']),('test','test',sorted(features['test']))]
    for name,source_split,ids in populations:
        b=branches[source_split];f=features[source_split];methods={};disc={}
        for family in plan['families']:
            scores={s:f[s][family]for s in ids};ranked=sorted_ids(scores,ids)
            disc[family]={str(q):discrimination([scores[s]for s in ids],[good(b[s]['light'],q)for s in ids])
                for q in cfg['quality_f1_thresholds']}
            methods[family]={}
            for fraction in cfg['bypass_fractions']:
                n=round(len(ids)*fraction);accepted=ranked[:n];point=evaluate(b,ids,accepted,.8)
                quality={str(q):sum(good(b[s]['light'],q)for s in accepted)
                    for q in cfg['quality_f1_thresholds']}
                methods[family][str(fraction)]={'requested_images':n,'quality_good_counts':quality,**point}
                for q,g in quality.items():
                    rows.append({'split':name,'family':family,'budget_fraction':fraction,'quality_target':q,
                        'accepted_images':len(accepted),'requested_images':n,'good_images':g,
                        'good_fraction':g/len(accepted)if accepted else None,
                        'accepted_word_f1':point['accepted_words']['f1'],
                        'rejected_deepsolo_f1':point['deepsolo_rejected_words']['f1'],
                        'final_f1':point['combined']['f1'],'tn0':point['combined']['accuracy_tn0'],
                        **{k:point['combined'][k]for k in ('tp','fp','fn','detected_words','gt_words')}})
        diagnostics['splits'][name]={'fixed_budgets':methods,'discrimination':disc}
    ids=sorted(features['test']);b=branches['test'];test=diagnostics['splits']['test']['fixed_budgets']
    rng=np.random.default_rng(cfg['seed']);weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),
        size=cfg['bootstrap_iterations']).astype(float)
    for budget in cfg['bypass_fractions']:
        key=str(budget);candidate=test['base_proxy_f1'][key];reference=test['all_checks_softcrop'][key]
        diagnostics['paired_intervals'][key]={
            'candidate':'base_proxy_f1','reference':'all_checks_softcrop',
            'delta_f1':candidate['combined']['f1']-reference['combined']['f1'],
            'ci95':paired_intervals(b,ids,candidate['accepted_ids'],reference['accepted_ids'],weights),
            'delta_vs_deepsolo':candidate['combined']['f1']-result['splits']['test']['deepsolo_only']['f1'],
            'ci95_vs_deepsolo':paired_intervals(b,ids,candidate['accepted_ids'],[],weights),
            'iterations':cfg['bootstrap_iterations'],'gates_fixed':True,'multiplicity_correction':False,
            'interpretation':'Exploratory fixed-gate uncertainty; no policy-selection uncertainty or multiple-test guarantee.'}
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2,sort_keys=True,allow_nan=False)+'\n')
    with(out/'all_method_budgets.csv').open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plot(out,result,diagnostics,cfg)
    report(out,result,diagnostics,cfg)
    print(json.dumps({'status':'SUCCESS','root':str(out),'descriptive_rows':len(rows),
        'diagnostics_sha256':sha(out/'diagnostics.json')}))


def plot(out,result,data,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16.5,4.8),layout='constrained')
    baseline=result['splits']['test']['deepsolo_only']['f1']*100
    oracle=result['splits']['test']['quality_oracles'];qs=cfg['quality_f1_thresholds']
    axes[0].plot([q*100 for q in qs],[oracle[str(q)]['combined']['f1']*100 for q in qs],
        'o-',color='tab:purple',label='GT-only quality gate (not deployable)')
    other=axes[0].twinx()
    other.plot([q*100 for q in qs],[oracle[str(q)]['accepted_images']/3 for q in qs],
        's--',color='tab:gray',label='Images retained (%)')
    other.set_ylabel('Images retained (%)',color='tab:gray');other.set_ylim(0,100)
    axes[0].set_xlabel('Required light-branch per-image F1 (%)')
    axes[0].set_title('A. Actual DeepSolo on the rejected subset')
    axes[0].legend(loc='lower right',fontsize=8)
    methods=[('coverage_baseline','Coverage'),('all_checks_softcrop','Crop + scale + coverage'),
        ('base_proxy_f1','New coverage/word proxy'),('proxy_f1_stable','New crop stability proxy'),
        ('paper_caption_control','Paper caption control'),('base_lm_coherence','Context LM control')]
    test=data['splits']['test']['fixed_budgets']
    for family,label in methods:
        axes[1].plot([b*100 for b in cfg['bypass_fractions']],
            [test[family][str(b)]['combined']['f1']*100 for b in cfg['bypass_fractions']],
            'o-',label=label,markersize=4)
        axes[2].plot([b*100 for b in cfg['bypass_fractions']],
            [test[family][str(b)]['good_fraction']*100 for b in cfg['bypass_fractions']],
            'o-',label=label,markersize=4)
    axes[1].set_title('B. Fixed budgets; methods declared before TEST')
    axes[2].set_title('C. Accepted images reaching F1 >= 80%')
    for ax in axes[:2]:
        ax.axhline(baseline,color='black',ls=':',label=f'All DeepSolo ({baseline:.2f})')
        ax.axhline(cfg['paper_nominal_targets']['ViTAE']*100,color='firebrick',ls='--',alpha=.6)
        ax.set_ylabel('Final local word F1 (%)')
    axes[1].set_xlabel('Images retained without DeepSolo (%)');axes[1].legend(fontsize=7,loc='lower left')
    axes[2].set_xlabel('Images retained without DeepSolo (%)');axes[2].set_ylabel('Accepted good-image fraction (%)')
    axes[2].axhline(90,color='firebrick',ls='--',label='90% empirical purity target')
    axes[2].set_ylim(35,100)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Total-Text TEST300: approximate lexical metric; no trained selector\n'
        'Red F1 line: paper nominal 79.8, protocol equivalence unresolved. TEST curves are descriptive.',fontsize=11)
    fig.savefig(out/'quality_study.png',dpi=180);fig.savefig(out/'quality_study.pdf');plt.close(fig)


def report(out,result,data,cfg):
    fmt=lambda v:'undefined'if v is None else f'{100*v:.2f}%'
    lines=['# Frozen F1-quality study: descriptive supplement','',
        'Approximate reproduction. No training, new inference or TEST-based deployment selection.',
        'Actual word counts are summed across retained light outputs and rejected DeepSolo outputs.',
        '','## GT-only quality gates (not deployable)','',
        '|Light per-image F1 target|TRAIN retained /125|TRAIN final F1|TEST retained /300|Accepted word F1|Rejected DeepSolo F1|TEST final F1|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for q in cfg['quality_f1_thresholds']:
        tr=result['splits']['train']['quality_oracles'][str(q)];te=result['splits']['test']['quality_oracles'][str(q)]
        lines.append(f"|{fmt(q)}|{tr['accepted_images']}|{fmt(tr['combined']['f1'])}|{te['accepted_images']}|{fmt(te['accepted_words']['f1'])}|{fmt(te['deepsolo_rejected_words']['f1'])}|{fmt(te['combined']['f1'])}|")
    lines+=['','## Equal 20% budget: fixed rules, descriptive comparison','',
        '|Method|TRAIN final F1|TEST good /60 at quality80%|Accepted word F1|Rejected DeepSolo F1|TEST final F1|TEST TN0|',
        '|---|---:|---:|---:|---:|---:|---:|']
    families=['coverage_baseline','all_checks_softcrop','base_proxy_f1','proxy_f1_stable',
        'paper_caption_control','base_lm_coherence','base_mpnet','base_clip_context',
        'base_vqa_visual_drop_plain']
    for family in families:
        tr=data['splits']['train']['fixed_budgets'][family]['0.2'];te=data['splits']['test']['fixed_budgets'][family]['0.2']
        lines.append(f"|{family}|{fmt(tr['combined']['f1'])}|{te['good_images']}|{fmt(te['accepted_words']['f1'])}|{fmt(te['deepsolo_rejected_words']['f1'])}|{fmt(te['combined']['f1'])}|{fmt(te['combined']['accuracy_tn0'])}|")
    lines+=['','## New base_proxy_f1 versus the previous consistency control','',
        '|Budget|TRAIN final F1|TEST accepted|TEST good fraction|TEST final F1|Delta versus control (pp)|Paired 95% interval (pp)|',
        '|---|---:|---:|---:|---:|---:|---|']
    for budget in cfg['bypass_fractions']:
        key=str(budget);tr=data['splits']['train']['fixed_budgets']['base_proxy_f1'][key]
        te=data['splits']['test']['fixed_budgets']['base_proxy_f1'][key];ci=data['paired_intervals'][key]
        lines.append(f"|{fmt(budget)}|{fmt(tr['combined']['f1'])}|{te['accepted_images']}|{fmt(te['good_fraction'])}|{fmt(te['combined']['f1'])}|{ci['delta_f1']*100:+.3f}|[{ci['ci95'][0]*100:+.3f}, {ci['ci95'][1]*100:+.3f}]|")
    lines+=['','The family is fixed for this diagnostic table; it was not selected on TEST.',
        'Threshold-calibrated deployment policies remain in ../report.md and ../results.json.',
        'No TEST-based ranking is a confirmed new deployment winner. Zero-bypass fallback is not routing success.',
        'The 90% good-image target is empirical and does not guarantee TEST purity.',
        'All source/test reuse, nonofficial metric, TRAIN fine-tuning and latency limitations still apply.',
        'Standalone local DeepSolo TEST F1: '+fmt(result['splits']['test']['deepsolo_only']['f1'])+'.',
        'All declared rules/budgets/splits/quality targets: all_method_budgets.csv and diagnostics.json.',
        'Paired intervals use 10,000 image bootstrap replicates, fixed gates, and no multiplicity correction.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--run',type=Path,required=True)
    analyze(parser.parse_args().run)
