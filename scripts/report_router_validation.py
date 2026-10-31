"""Export audited validation-calibrated router results and cause controls."""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

from scripts.run_yolo_f1_quality_study import read,seal,sha


def cost(p):
    return f"{p['deepsolo_images']}/{p['total_evaluated_images']} ({p['deepsolo_percentage']:.2f}%)"


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('run-root','controls','gt-control','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--localization-control',type=Path)
    args=parser.parse_args();root=args.run_root;out=args.output
    d=read(root/'results.json');plan=read(root/'resolved_configuration.json');frozen=read(root/'frozen_choices.json');audit=read(root/'independent_audit.json');completion=read(root/'completion.json')
    if (d['reproduction_mode']!='approximate' or audit['status']!='PASS' or completion['status']!='SUCCESS'
        or sha(root/'results.json')!=completion['results_sha256'] or sha(root/'independent_audit.json')!=completion['audit_sha256']
        or sha(root/'frozen_choices.json')!=d['frozen_choices_sha256']):raise ValueError('Completed audited validation study required')
    control=read(args.controls);gt=read(args.gt_control);primary=str(d['primary_seed'])
    rows=[];lines=['# Generalization controls and unchanged-weight validation calibration','',
        '**Approximate reproduction.** Fit1128 / validation125 / TEST300. '
        'Epoch and thresholds use validation only; deployment uses identical selected weights. '
        'There is no refit after calibration. The first fixed seed is primary; '
        'remaining seeds are variance diagnostics, not TEST-based candidates.', '',
        'Primary F1 is Unicode casefold+alphanumeric per-image lexical multiset micro F1, '
        'without localization IoU. Original care, ignored-output and empty-string denominators '
        'are retained. Costs below are logical routing on cached experts; actual new '
        'DeepSolo inference is0. Historical validation and TEST exposure remain.', '',
        '## Expert training exposure controls','',
        '| Population | Expert | F1 (%) | DeepSolo images/total (%) |',
        '| --- | --- | ---: | ---: |']
    for population,value in control['expert_exposure_endpoints'].items():
        for name in ('unet','yolo','deepsolo'):
            p=value[name]
            lines.append(f"| {population} | {name.upper()} | {100*p['metrics']['f1']:.4f} | {p['deepsolo_images']}/{p['total_images']} ({p['deepsolo_percentage']:.2f}%) |")
    lines+=['','## Recognition with GT crops','',
        'This uses annotations to supply perfect rectangles and measures paired word '
        'recognition, not deployable routing or primary lexical F1. The125 fitting '
        'control images match validation GT word-count strata. Every care word remains.', '',
        '| Population | Correct/GT words | Paired accuracy (%) | DeepSolo images/total (%) |',
        '| --- | ---: | ---: | ---: |']
    for population,value in gt['results'].items():
        lines.append(f"| {population} | {value['correct_words']}/{value['gt_words']} | {100*value['paired_recognition_accuracy']:.4f} | 0/{value['images']} (0.00%) |")
    if args.localization_control:
        localization=read(args.localization_control)
        lines+=['','## Localization and recognition error partition','',
            'Diagnostic one-to-one spatial matching uses bbox IoU>=0.5. It is '
            'separate from the primary lexical F1. Uncovered words include missed, '
            'merged or shifted envelopes. Nested maximum-cardinality counts '
            'partition every GT word without assigning a primary lexical error cause.', '',
            '| Population | GT words | Raw crop geometric coverage | No geometric coverage | Covered without correct reading | Correct reading removed by filter | Retained spatial correct | DeepSolo images/total (%) |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for name,v in localization['cohorts'].items():
            c=v['counts']
            lines.append(f"| {name} | {c['gt_words']} | {100*c['geometry_covered_fraction']:.4f}% | {c['no_geometric_coverage']} | {c['covered_but_no_correct_reading']} | {c['correct_reading_removed_by_filter']} | {c['retained_spatial_correct']} | 0/{v['images']} (0.00%) |")
    lines+=['','## OCR filter controls','',
        'The source OCR filter0.7 is retained in the primary experiment. These are '
        'predeclared diagnostic sweeps, with no TEST-based change.', '',
        '| Sequence cut | Fit F1 (DS0/1128,0%) | Validation F1 (DS0/125,0%) | TEST F1 (DS0/300,0%) |',
        '| ---: | ---: | ---: | ---: |']
    for threshold in control['ocr_filter_controls']['fit']:
        values=[100*control['ocr_filter_controls'][s][threshold]['metrics']['f1'] for s in ('fit','validation','test')]
        lines.append(f'| {threshold} | {values[0]:.4f}% | {values[1]:.4f}% | {values[2]:.4f}% |')
    lines+=['','## Primary scalar thresholds','',
        'Targets below refer to acceptance on validation. Actual TEST acceptance may '
        'differ. The threshold is a predicted-utility score, not a probability.', '']
    for split,models in d['splits'].items():
        lines += [f'### {split.upper()}', '',
            '| Expert | Validation target | Cut | Accepted light | DeepSolo images/total (%) | Hybrid F1 (%) |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for model,seeds in models.items():
            for seed,value in seeds.items():
                for kind,points in [('VALIDATION_TRANSFER',value['thresholds']),('DESCRIPTIVE_QUOTA',list(value['descriptive_equal_quotas'].values()))]:
                    for index,p in enumerate(points):
                        n=p['total_evaluated_images'];m=p['combined']
                        if p['deepsolo_images']+p['accepted_images']!=n or abs(p['deepsolo_percentage']-100*p['deepsolo_images']/n)>1e-10:
                            raise ValueError('Report routing cost mismatch')
                        if abs(m['f1']-2*m['tp']/(2*m['tp']+m['fp']+m['fn']))>1e-12:raise ValueError('Report F1 mismatch')
                        policy=p.get('policy',{});target=policy.get('validation_target_fraction',plan['config']['validation_study']['acceptance_targets'][index])
                        cut=policy.get('threshold')
                        rows.append(dict(split=split,model=model,seed=seed,primary=seed==primary,kind=kind,
                            validation_target=target,threshold=cut,accepted_images=p['accepted_images'],
                            deepsolo_images=p['deepsolo_images'],total_images=n,deepsolo_percentage=p['deepsolo_percentage'],
                            f1_percent=100*m['f1'],accuracy_tn0_percent=100*m['accuracy_tn0'],
                            tp=m['tp'],fp=m['fp'],fn=m['fn'],predicted_words=m['detected_words'],gt_words=m['gt_words']))
                        if kind=='VALIDATION_TRANSFER' and seed==primary:
                            text='reject all' if cut is None else f'{cut:.6f}'
                            lines.append(f"| {model.upper()} | {100*target:.0f}% | {text} | {p['accepted_images']}/{n} | {cost(p)} | {100*m['f1']:.4f} |")
        lines.append('')
    lines+=['## Seed controls at the validation50% cut','',
        '| Expert | Seed | Selected epoch | TEST accepted | TEST DeepSolo images/total (%) | TEST F1 (%) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for model,seeds in d['splits']['test'].items():
        for seed,value in seeds.items():
            p=next(p for p in value['thresholds'] if p['policy']['validation_target_fraction']==.5)
            epoch=frozen['choices'][model][seed]['epoch']
            lines.append(f"| {model.upper()} | {seed}{' primary' if seed==primary else ''} | {epoch} | {p['accepted_images']}/300 | {cost(p)} | {100*p['combined']['f1']:.4f} |")
    lines+=['','## Interpretation limits','',
        'Expert fitting versus holdout differences remain within word-count and '
        'relative-text-height strata; the JSON controls preserve every bin and its '
        'image counts. These controls weaken a pure density/small-text explanation '
        'without isolating all typography or scene differences.', '',
        'Original OOF and full-refit count predictions are compared on identical '
        'images and targets in the controls JSON. This measures raw-score bias '
        'after changing the model. The corrected protocol calibrates each chosen '
        'model on its held-out validation predictions and verifies exact score '
        'replay when loading that checkpoint.', '',
        'Validation was historically reused for expert checkpoint/profile selection. '
        'Router fitting still uses expert-fitting images; fully removing this '
        'expert-exposure difference would require out-of-fold expert predictions '
        'or additional expert-unseen training images. Seed repeats diagnose variance '
        'but do not provide an independent new TEST benchmark.', '']
    out.mkdir(parents=True,exist_ok=True)
    with (out/'report.md').open('x') as stream:stream.write('\n'.join(lines))
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/'validation_thresholds.csv').open('x') as stream:stream.write(buffer.getvalue())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for axis,split in zip(axes,('validation','test')):
        n=len(plan['partitions'][split])
        for model,color in [('unet','#2563eb'),('yolo','#e8790c')]:
            points=d['splits'][split][model][primary]['thresholds']
            axis.plot([p['deepsolo_images'] for p in points],[100*p['combined']['f1'] for p in points],'-o',color=color,label=model.upper())
        base=d['splits'][split]['unet'][primary]['all_deepsolo']['metrics']['f1']
        axis.axhline(100*base,color='#111827',ls=':',label=f'All DeepSolo {n}/{n} (100%)')
        ticks=sorted({round(n*q) for q in [0,.25,.5,.75,1]})
        axis.set_xticks(ticks,[f'{t}/{n}\n{100*t/n:.1f}%' for t in ticks]);axis.set_xlabel('DeepSolo images / total (%) — cached logical routing')
        axis.set_ylabel('Alphanumeric lexical micro F1 (%)');axis.set_title(split.upper()+' — validation cuts, same deployed weights')
        axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.suptitle('Approximate reproduction: validation-selected thresholds, no refit')
    fig.savefig(out/'validation_curves.png',dpi=160);fig.savefig(out/'validation_curves.svg');plt.close(fig)
    seal(out/'report_receipt.json',dict(status='PASS',results_sha256=sha(root/'results.json'),code_sha256=sha(__file__),
        controls_sha256=sha(args.controls),gt_control_sha256=sha(args.gt_control),csv_rows=len(rows),
        localization_control_sha256=sha(args.localization_control) if args.localization_control else None,
        files_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(out/'report.md')


if __name__=='__main__':main()
