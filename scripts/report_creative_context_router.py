"""Export all six context methods, validation-only group choices and diagnostics."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.run_yolo_f1_quality_study import checked,read,seal,sha


def report(root):
    plan=read(root/'resolved_configuration.json');cfg=plan['config'];result=read(root/'results.json');frozen=read(root/'frozen_choices.json')
    audit=read(root/'independent_audit.json')
    if audit['status']!='PASS' or audit['results_sha256']!=sha(root/'results.json'):raise ValueError('A completed independent audit is required')
    names=plan['families'];groups={s:['ridge_'+s,'trees_'+s] for s in cfg['feature_sets']}
    groups['memory']=['memory_'+v for v in cfg['memory_variants']]
    reference_names={'0.2':'base_proxy_f1','0.3':'risk_visual_consensus','0.5':'risk_visual_mass'}
    branches_path=Path(plan['source_coverage_root'])/'results.json'
    branches=checked(branches_path,plan['input_sha256'][str(branches_path)])['resolutions']['640']['branch_metrics']
    ids=sorted(branches['test']);rng=np.random.default_rng(cfg['execution']['seed'])
    weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=cfg['bootstrap_iterations']).astype(float)
    def distribution(accepted):
        a=set(accepted);rows=np.array([[branches['test'][s]['light' if s in a else 'deepsolo'][k] for k in ('tp','fp','fn')] for s in ids])
        c=weights@rows;return 2*c[:,0]/(2*c[:,0]+c[:,1]+c[:,2])
    summary={};all_ds=distribution([])
    for budget,data in result['splits']['test']['budgets'].items():
        val=frozen['budgets'][budget]['validation_f1'];ref_name=reference_names[budget];ref=data['controls'][ref_name]
        ref_boot=distribution(ref['accepted_ids']);rows={}
        for group,families in groups.items():
            family=max(families,key=lambda f:(val[f],-names.index(f)));p=data['families'][family]
            boot=distribution(p['accepted_ids']);dif=boot-ref_boot
            rows[group]={'family_chosen_using_validation_only':family,
                'train':result['splits']['train']['budgets'][budget]['families'][family],
                'validation':result['splits']['validation']['budgets'][budget]['families'][family],'test':p,
                'delta_f1_vs_historical_control':p['combined']['f1']-ref['combined']['f1'],
                'paired_ci95_vs_historical_control':np.quantile(dif,[.025,.975]).tolist(),
                'paired_ci95_vs_all_deepsolo':np.quantile(boot-all_ds,[.025,.975]).tolist()}
        summary[budget]={'groups':rows,'selected_family':data['selected_family'],'selected_test':data['selected'],
            'historical_control':ref_name,'historical_control_test':ref}
    coverage={}
    for split,ss in plan['samples'].items():
        features=[read(root/split/'features'/(s['sample_id'].replace(':','__')+'.json')) for s in ss]
        records={'images':len(ss),'images_with_glyph_pairs':0,'images_with_rendering':0,'images_with_structure':0,
            'glyph_comparison_pairs':0,'rendered_words':0,'conditional_abstentions':0,'predicted_words':0,'swap_words':0,
            'attention_guided_character_windows':0}
        labels=[]
        for sample,f in zip(ss,features):
            evidence=f['evidence'];v=f['values'];records['images_with_glyph_pairs']+=not bool(v['glyph_missing'])
            records['images_with_rendering']+=not bool(v['render_margin_missing']);records['images_with_structure']+=not bool(v['structure_missing'])
            records['glyph_comparison_pairs']+=len(evidence['glyph']['pairs'])
            records['rendered_words']+=sum('abstention' not in e for e in evidence['render'])
            key=sample['sample_id'].replace(':','__')+'.json';perm=read(root/split/'permutation'/key)
            records['predicted_words']+=len(perm['words']);records['conditional_abstentions']+=sum(bool(w.get('abstention')) for w in perm['words'])
            records['attention_guided_character_windows']+=sum(len(w.get('characters',[])) for w in perm['words'])
            records['swap_words']+=len(read(root/split/'swap'/key)['probes'])
            c=branches[split][sample['sample_id']]['light'];den=2*c['tp']+c['fp']+c['fn']
            labels.append(den>0 and Fraction(2*c['tp'],den)>=Fraction(str(cfg['quality_f1_threshold'])))
        # Applicable-only diagnostic populations are reported; final F1 retains all images/words.
        signals={'swap':('swap_foreign_evidence_mean','swap_foreign_evidence_missing',1),
            'permutation':('permutation_support_minimum','permutation_support_missing',1),
            'glyph':('glyph_contradiction','glyph_missing',-1),'render':('render_margin_mean','render_margin_missing',1)}
        auroc={}
        for name,(field,missing,direction) in signals.items():
            take=[i for i,f in enumerate(features) if not f['values'][missing]]
            yy=[labels[i] for i in take];xx=[direction*features[i]['values'][field] for i in take]
            auroc[name]={'applicable_images':len(take),'good80_images':sum(yy),
                'auroc':float(roc_auc_score(yy,xx)) if len(set(yy))==2 else None}
        records['quality80_auroc_applicable_only_descriptive']=auroc;coverage[split]=records
    out={'claim_label':'approximate reproduction','status':'AUDITED_EXPLORATORY_RESULTS','budgets':summary,'coverage':coverage,
        'group_choices_use_test_labels':False,'bootstrap_gates_fixed':True,'multiplicity_correction':False,
        'historical_reference_is_exploratory':True,'combined_contains_five_raw_signal_groups_memory_is_separate':True,
        'plan_sha256':sha(root/'resolved_configuration.json'),'results_sha256':sha(root/'results.json'),
        'audit_sha256':sha(root/'independent_audit.json'),'report_code_sha256':sha(__file__),
        'software':{n:importlib.metadata.version(n) for n in ('numpy','scikit-learn','matplotlib')},'python':sys.version}
    seal(root/'analysis_summary.json',out)
    rows=[]
    for split,data in result['splits'].items():
        for budget,value in data['budgets'].items():
            for kind in ('families','scalar_thresholds'):
                for family,p in value[kind].items():
                    rows.append({'split':split,'budget':budget,'kind':kind,'family':family,'validation_selected':family==value['selected_family'],
                        'accepted_images':p['accepted_images'],'good80_images':p['good_images'],
                        **{k:p['combined'][k] for k in ('tp','fp','fn','detected_words','gt_words','all_outputs','ignored_outputs','f1','accuracy_tn0','precision','recall')}})
    with (root/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    budgets=sorted(summary,key=float);lines=['# Six context signals: approximate reproduction','','Exploratory word-multiset microF1, not paper accuracy.','',
        '|Bypass|Frozen VAL choice|TRAIN125 F1|TEST300 F1|Historical control TEST F1|',
        '|---|---|---:|---:|---:|']
    for b in budgets:
        family=summary[b]['selected_family'];train=result['splits']['train']['budgets'][b]['families'][family]
        lines.append(f'|{100*float(b):.0f}%|{family}|{100*train["combined"]["f1"]:.3f}%|{100*summary[b]["selected_test"]["combined"]["f1"]:.3f}%|{100*summary[b]["historical_control_test"]["combined"]["f1"]:.3f}%|')
    lines+=['','Per-method variants below are selected using frozen validation F1 within each group.','',
        '|Method|TEST20% F1|TEST30% F1|TEST50% F1|','|---|---:|---:|---:|']
    for group in groups:
        lines.append('|'+group+'|'+'|'.join(f'{100*summary[b]["groups"][group]["test"]["combined"]["f1"]:.3f}%' for b in budgets)+'|')
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for group in groups:
        yy=[100*summary[b]['groups'][group]['test']['combined']['f1'] for b in budgets]
        ax.plot([100*float(b) for b in budgets],yy,marker='o',label=group)
    ax.axhline(100*result['splits']['test']['deepsolo_only']['f1'],linestyle='--',color='black',label='all DeepSolo')
    ax.set(xlabel='Images bypassing DeepSolo (%)',ylabel='Final word micro F1 (%)',title='Six context methods — exploratory TEST300')
    ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2);fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(root/('creative_context_comparison.'+suffix),dpi=180)
    plt.close(fig)
    print(json.dumps({'status':'SUCCESS','report':str(root/'report.md'),'summary_sha256':sha(root/'analysis_summary.json'),'coverage':coverage}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--run',type=Path,required=True);report(parser.parse_args().run.resolve())
