"""Export audited matched U-Net/YOLO router curves without rerunning inference."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import io
from pathlib import Path
import sys

import numpy as np

from scripts.run_yolo_f1_quality_study import read, seal, sha


def verify_frozen_evidence(plan, root):
    """Recheck source, stage and checkpoint bytes after the completed run."""
    count=0
    for name, digest in {**plan['inputs_sha256'], **plan['assets_sha256']}.items():
        if sha(name)!=digest:
            raise ValueError('Frozen report input changed: '+name)
        count+=1
    for role, entries in plan['config']['source_sha256'].items():
        for name, digest in entries.items():
            if sha(Path(plan['paths'][role])/name)!=digest:
                raise ValueError('Qualified source report changed: '+role+'/'+name)
            count+=1
    for name,digest in plan['code_sha256'].items():
        if sha(root/'code_snapshot'/name)!=digest:
            raise ValueError('Executed source snapshot changed: '+name)
        count+=1
    for split in ('train','test'):
        for stage in ('unet','fallback','features'):
            value=read(root/split/(stage+'_receipt.json'))
            if value['plan_sha256']!=sha(root/'resolved_configuration.json'):
                raise ValueError('Stage report plan changed')
            for name,digest in value['files_sha256'].items():
                if sha(root/name)!=digest:
                    raise ValueError('Generated stage evidence changed: '+name)
                count+=1
    frozen=read(root/'frozen_choices.json')
    for value in [*frozen['choices'].values(), *frozen['cv_models'].values()]:
        if sha(root/value['model_path'])!=value['model_sha256']:
            raise ValueError('Learned checkpoint changed after independent audit')
        count+=1
    return dict(status='PASS', verified_sha256_items=count,
                scope='Qualified external inputs, immutable executed source, generated stage receipts and learned checkpoints')


def route_rows(results):
    """Keep deployment cuts, descriptive quotas and standalone costs explicit."""
    rows = []
    for split, models in results['splits'].items():
        for backbone, value in models.items():
            n = value['all_light']['images']
            policies = [dict(kind='standalone_light', threshold=None, target=None,
                             accepted_images=n, deepsolo_images=0, total_evaluated_images=n,
                             deepsolo_percentage=0., combined=value['all_light']),
                        dict(kind='all_deepsolo', threshold=None, target=None,
                             accepted_images=0, deepsolo_images=n, total_evaluated_images=n,
                             deepsolo_percentage=100., combined=value['all_deepsolo'])]
            for point in value['scalar_threshold_results']:
                p = point['policy']
                policies.append(dict(point, kind=p['kind'], threshold=p['threshold'],
                                     target=p.get('train_target_fraction')))
            for target, point in value['equal_quota_results'].items():
                policies.append(dict(point, kind='descriptive_equal_quota', threshold=None,
                                     target=float(target)))
            for p in policies:
                if (p['total_evaluated_images'] != n or p['accepted_images']+p['deepsolo_images'] != n
                        or abs(p['deepsolo_percentage']-100*p['deepsolo_images']/n) > 1e-10):
                    raise ValueError('Reported routing cost does not match evaluated population')
                metrics = p['combined']
                rows.append(dict(split=split, backbone=backbone, kind=p['kind'],
                    target_acceptance=p['target'], threshold=p['threshold'],
                    accepted_images=p['accepted_images'], deepsolo_images=p['deepsolo_images'],
                    total_images=n, deepsolo_percentage=p['deepsolo_percentage'],
                    **{k:metrics[k] for k in ('tp','fp','fn','detected_words','gt_words')},
                    f1_percent=100*metrics['f1'], accuracy_tn0_percent=100*metrics['accuracy_tn0'],
                    precision_percent=100*metrics['precision'] if metrics['precision'] is not None else None,
                    recall_percent=100*metrics['recall'] if metrics['recall'] is not None else None))
    return rows


def cost(point):
    return f"{point['deepsolo_images']}/{point['total_evaluated_images']} ({point['deepsolo_percentage']:.2f}%)"


def paired_quota_intervals(plan, root, results):
    """Paired RGB-cluster bootstrap, conditional on already frozen TEST routes."""
    data = read(root/'test/branches.json')['branches']
    samples = {s['sample_id']:s for s in plan['samples']['test']}
    ids = sorted(samples); unique = sorted({s['rgb_sha256'] for s in samples.values()})
    mapping = np.array([unique.index(samples[s]['rgb_sha256']) for s in ids])
    rng = np.random.default_rng(plan['config']['execution']['seed'])
    weights = np.array([np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[mapping]
                       for _ in range(plan['config']['bootstrap_replicates'])], float)
    intervals = {}
    for target in results['splits']['test']['unet']['equal_quota_results']:
        draws = {}
        for model in ('unet','yolo'):
            point = results['splits']['test'][model]['equal_quota_results'][target]
            selected = set(point['accepted_ids'])
            a = np.array([[data[model][s]['light' if s in selected else 'deepsolo'][k]
                           for k in ('tp','detected_words','gt_words')] for s in ids], float)
            if not np.array_equal(a.sum(0), [point['combined'][k] for k in ('tp','detected_words','gt_words')]):
                raise ValueError('Report bootstrap inputs do not replay the frozen route counts')
            a = weights@a
            draws[model] = 200*a[:,0]/(a[:,1]+a[:,2])
        low, high = np.quantile(draws['unet']-draws['yolo'], [.025,.975])
        intervals[target] = dict(lower95_pp=float(low), upper95_pp=float(high),
            conditional_on_fixed_routes=True, includes_training_variance=False)
    return intervals


def render_markdown(plan, results, choices, intervals):
    population = ('Smoke-test subsets only; no performance claim.' if plan['result_class']=='SMOKE_TEST'
                  else 'Complete usable TRAIN and TEST are separate.')
    lines = ['# Matched token routing: qualified U-Net versus YOLO', '',
        '**Approximate reproduction.** '+population+' '
        'Word-multiset micro F1 uses Unicode casefold and alphanumeric characters; '
        'it does not require localization IoU and is not the paper\'s recognition accuracy.', '',
        'U-Net: qualified Text+Kernel control continuation epoch2, frozen PARSeq and '
        'TRAIN-selected sequence threshold0.7. YOLO: qualified frozen source pipeline. '
        'Both token heads use the same architecture, grouped folds, seed, optimizer '
        'and improved native ViTAEv2-S/TextOCR DeepSolo0.35/0.925 target.', '',
        'All routing costs below are logical decisions on cached expert outputs; '
        'they do not measure online inference latency. Context encoders and both '
        'OCR experts remain frozen. Only the small token head is trained.', '',
        f"Selected router epochs (TRAIN OOF): U-Net {choices['choices']['unet']['epoch']}; "
        f"YOLO {choices['choices']['yolo']['epoch']}.", '']
    for split, models in results['splits'].items():
        n = models['unet']['all_light']['images']
        lines += [f'## {split.upper()} ({n} images)', '',
            'TRAIN values use grouped out-of-fold scores; TEST uses heads refitted '
            'on the full usable TRAIN population.', '',
            '| Standalone endpoint | F1 (%) | DeepSolo images/total (%) |',
            '| --- | ---: | ---: |']
        for model in ('unet','yolo'):
            lines.append(f"| {model.upper()} + PARSeq | {100*models[model]['all_light']['f1']:.4f} | 0/{n} (0.00%) |")
        lines += [f"| Improved DeepSolo | {100*models['unet']['all_deepsolo']['f1']:.4f} | {n}/{n} (100.00%) |", '',
            '### Scalar thresholds selected on TRAIN OOF and transferred unchanged', '',
            'The target fraction identifies the TRAIN quantile, not guaranteed TEST acceptance. '
            'Ties are retained. Higher scores favor the light expert.', '',
            '| Backbone | TRAIN target | Actual cutoff | Accepted light | DeepSolo images/total (%) | F1 (%) |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for model in ('unet','yolo'):
            for p in models[model]['scalar_threshold_results']:
                policy = p['policy']
                if policy['kind'] != 'TRAIN_OOF_quantile':
                    continue
                cut = 'reject all' if policy['threshold'] is None else f"{policy['threshold']:.6f}"
                lines.append(f"| {model.upper()} | {100*policy['train_target_fraction']:.0f}% | {cut} | "
                    f"{p['accepted_images']}/{n} | {cost(p)} | {100*p['combined']['f1']:.4f} |")
        lines += ['', '### Descriptive comparison at equal light-image quotas', '',
            'This ranks each batch without using its labels. These are separate from '
            'the transferred scalar thresholds. Images without eligible nonempty readings '
            'always use DeepSolo, so large quotas can be capped. The oracle uses GT and is diagnostic.', '',
            '| Light quota | U-Net accepted | U-Net DeepSolo images/total (%) | U-Net F1 (%) | '
            'YOLO accepted | YOLO DeepSolo images/total (%) | YOLO F1 (%) | '
            'U-Net oracle F1 (%) | YOLO oracle F1 (%) |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for target, a in models['unet']['equal_quota_results'].items():
            b = models['yolo']['equal_quota_results'][target]
            lines.append(f"| {100*float(target):.0f}% | {a['accepted_images']}/{n} | {cost(a)} | {100*a['combined']['f1']:.4f} | "
                f"{b['accepted_images']}/{n} | {cost(b)} | {100*b['combined']['f1']:.4f} | "
                f"{100*a['oracle']['f1']:.4f} | {100*b['oracle']['f1']:.4f} |")
        if split=='test':
            lines += ['', 'Paired U-Net minus YOLO intervals, conditional on the fixed TEST routes '
                '(RGB-cluster bootstrap; training variance excluded):', '',
                '| Light quota | U-Net DeepSolo images/total (%) | YOLO DeepSolo images/total (%) | '
                'Difference F1 (pp) | 95% interval (pp) |', '| --- | ---: | ---: | ---: | ---: |']
            for target, ci in intervals.items():
                a=models['unet']['equal_quota_results'][target]; b=models['yolo']['equal_quota_results'][target]
                delta=100*(a['combined']['f1']-b['combined']['f1'])
                lines.append(f"| {100*float(target):.0f}% | {cost(a)} | {cost(b)} | {delta:.4f} | "
                             f"[{ci['lower95_pp']:.4f}, {ci['upper95_pp']:.4f}] |")
        lines.append('')
    lines += ['## Protocol limits', '',
        'F1 = 2TP/(2TP+FP+FN); TN0 accuracy = TP/(TP+FP+FN). All care GT remain '
        'in denominators. Duplicate words use per-image multiset counts. Empty normalized '
        'strings cannot match. Source native256 validity determines ignored outputs; '
        'there is no localization IoU in this primary metric.', '',
        'Historical TEST exploration, inherited expert training exposure and one training seed '
        'limit generalization claims. Excluding the two known cross-split RGB duplicates '
        'from router fitting does not erase their possible influence on the frozen experts. '
        'Neither paper accuracy equivalence nor strict reproduction is established.', '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); root=args.run_root.resolve(); output=args.output.resolve()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    completion=read(root/'completion.json'); results=read(root/'results.json')
    audit=read(root/'independent_audit.json'); plan=read(root/'resolved_configuration.json')
    if (completion['status']!='SUCCESS' or audit['status']!='PASS'
            or completion['results_sha256']!=sha(root/'results.json')
            or completion['audit_sha256']!=sha(root/'independent_audit.json')
            or audit['results_sha256']!=sha(root/'results.json')
            or results['plan_sha256']!=sha(root/'resolved_configuration.json')
            or results['frozen_choices_sha256']!=sha(root/'frozen_choices.json')):
        raise ValueError('A completed independently audited experiment is required')
    receipt=output/'report_receipt.json'
    if receipt.exists():
        old=read(receipt)
        if old['results_sha256']!=sha(root/'results.json') or old['report_code_sha256']!=sha(__file__):
            raise ValueError('Immutable report source changed')
        if any(sha(output/name)!=digest for name,digest in old['files_sha256'].items()):
            raise ValueError('Derived report output changed')
        print(output/'report.md'); return
    evidence=verify_frozen_evidence(plan,root)
    output.mkdir(parents=True,exist_ok=True)
    rows=route_rows(results); intervals=paired_quota_intervals(plan,root,results)
    buffer=io.StringIO(); writer=csv.DictWriter(buffer,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    with (output/'threshold_results.csv').open('x') as stream:
        stream.write(buffer.getvalue())
    choices=read(root/'frozen_choices.json')
    with (output/'report.md').open('x') as stream:
        stream.write(render_markdown(plan,results,choices,intervals))
    seal(output/'paired_quota_intervals.json',dict(results_sha256=sha(root/'results.json'),intervals=intervals))
    fig, axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for axis, split in zip(axes,('train','test')):
        models=results['splits'][split]; n=models['unet']['all_light']['images']
        for model, color in [('unet','#2563eb'),('yolo','#e8790c')]:
            points=list(models[model]['equal_quota_results'].values())
            xx=[p['deepsolo_images'] for p in points]
            axis.plot(xx,[100*p['combined']['f1'] for p in points],'-o',color=color,label=model.upper()+' token router',ms=4)
            axis.plot(xx,[100*p['oracle']['f1'] for p in points],'--',color=color,alpha=.65,label=model.upper()+' GT oracle')
        axis.axhline(100*models['unet']['all_deepsolo']['f1'],color='#111827',ls=':',label='All DeepSolo')
        ticks=sorted({round(n*q) for q in [0,.25,.5,.75,1]})
        axis.set_xticks(ticks,[f'{v}/{n}\n{100*v/n:.1f}%' for v in ticks])
        axis.set_title(split.upper()+(' grouped OOF' if split=='train' else ' frozen full-TRAIN heads'))
        axis.set_xlabel('Images sent to DeepSolo / total (%) — cached logical routing')
        axis.set_ylabel('Alphanumeric word-multiset micro F1 (%)')
        axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.suptitle(('Smoke test: ' if plan['result_class']=='SMOKE_TEST' else 'Approximate reproduction: ')
                 +'equal-quota U-Net versus YOLO token routing')
    fig.savefig(output/'routing_curves.png',dpi=160)
    fig.savefig(output/'routing_curves.svg'); plt.close(fig)
    files={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    seal(receipt,dict(status='PASS',results_sha256=sha(root/'results.json'),
        report_code_sha256=sha(__file__),files_sha256=files,csv_rows=len(rows),
        source_run=str(root),reproduction_mode='approximate',routing_scope='cached logical decisions',
        evidence_recheck=evidence,
        report_runtime=dict(python=sys.version, executable=sys.executable,
                            packages={n:importlib.metadata.version(n) for n in ('numpy','matplotlib')})))
    print(output/'report.md')


if __name__=='__main__':
    main()
