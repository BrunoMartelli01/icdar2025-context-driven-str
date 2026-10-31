"""Descriptive signal diagnostics and scientific plots for an audited frozen run."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.run_yolo_f1_quality_study import checked, read, seal, sha


def report(root):
    plan=read(root/'resolved_configuration.json');result=read(root/'results.json')
    audit=read(root/'independent_audit.json')
    if audit['status']!='PASS' or audit['results_sha256']!=sha(root/'results.json'):
        raise ValueError('A completed independent audit is required')
    branches_path=Path(plan['source_coverage_root'])/'results.json'
    branches=checked(branches_path,plan['input_sha256'][str(branches_path)])['resolutions']['640']['branch_metrics']
    diagnostics={}
    for split in ('train','test'):
        rows=[read(root/split/'features'/(s['sample_id'].replace(':','__')+'.json')) for s in plan['samples'][split]]
        yes=[];erase=[];blank=[];foils=[];quality=[]
        for row in rows:
            value=read(root/split/'verification'/(row['sample_id'].replace(':','__')+'.json'))
            for probe in value['probes']:
                scores=probe['answers'];yes.append(scores['local'][0]['yes_probability'])
                foils.append(scores['local'][1]['yes_probability']);erase.append(scores['erased'][0]['yes_probability'])
                blank.append(scores['blank'][0]['yes_probability'])
            c=branches[split][row['sample_id']]['light'];den=2*c['tp']+c['fp']+c['fn']
            quality.append(den>0 and Fraction(2*c['tp'],den)>=Fraction(str(plan['config']['quality_f1_threshold'])))
        # Risk direction is its semantic definition; no TEST direction/threshold search.
        fields={'inventory_uncovered_fraction':-1,'inventory_lexical_disagreement':-1,
            'inventory_missing_word_proxy':-1,'verification_visual_margin_mean':1,
            'verification_support_mean':1,'control_base_proxy_f1':1}
        auc={name:float(roc_auc_score(quality,[sign*a['values'][name] for a in rows])) for name,sign in fields.items()}
        diagnostics[split]={'images':len(rows),'good80':sum(quality),
            'ocr_incomplete':sum(a['values']['inventory_ocr_incomplete'] for a in rows),
            'objects_incomplete':sum(a['values']['inventory_objects_incomplete'] for a in rows),
            'empty_inventory':sum(a['values']['inventory_empty'] for a in rows),'verified_words':len(yes),
            'median_yes':float(np.median(yes)),'median_foil_yes':float(np.median(foils)),
            'median_erased_yes':float(np.median(erase)),'median_blank_yes':float(np.median(blank)),
            'quality80_auroc_descriptive_only':auc}
    out={'status':'DESCRIPTIVE_ONLY','used_to_select_or_fit':False,'all_images_included':True,
        'missing_controls':'Reuse zero-filled scalar inputs and their missing indicators; AUROC uses the scalar',
        'quality_definition':'Exact per-image word-multiset F1 >= configured0.8; undefined empty denominator is bad',
        'not_the_optimization_target':'Final aggregate branch microF1, rather than good80 classification',
        'diagnostics':diagnostics,'plan_sha256':sha(root/'resolved_configuration.json'),
        'results_sha256':sha(root/'results.json'),'audit_sha256':sha(root/'independent_audit.json'),'code_sha256':sha(__file__)}
    seal(root/'signal_diagnostics.json',out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    budgets=sorted(result['splits']['test']['budgets'],key=float);fig,ax=plt.subplots(figsize=(7.2,4.5))
    labels={'core':'Previous scalar inputs','inventory':'+ Independent inventory','verified':'+ Inventory and local verification',
            'visual':'+ Frozen visual representation','combined':'+ All new evidence'}
    for variant,label in labels.items():
        values=[100*result['splits']['test']['budgets'][b]['families']['trees_'+variant]['combined']['f1'] for b in budgets]
        ax.plot([100*float(b) for b in budgets],values,marker='o',label=label)
    ax.axhline(100*result['splits']['test']['deepsolo_only']['f1'],color='black',linestyle='--',label='All DeepSolo')
    ax.set(xlabel='Images bypassing DeepSolo (%)',ylabel='Final word micro F1 (%)',
        title='Frozen ExtraTrees router ablations — exploratory TEST300')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
    for suffix in ('png','pdf'): fig.savefig(root/('grounded_context_comparison.'+suffix),dpi=180)
    plt.close(fig)
    print(json.dumps(out),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--run',type=Path,required=True)
    report(parser.parse_args().run.resolve())
