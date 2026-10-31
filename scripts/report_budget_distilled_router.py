"""Audited half-budget router comparison, validation cuts and CPU cost evidence."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

from scripts.run_yolo_f1_quality_study import read,seal,sha


def cost(p):return f"{p['deepsolo_images']}/{p['total_evaluated_images']} ({p['deepsolo_percentage']:.2f}%)"


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=args.run_root;out=args.output;out.mkdir(parents=True,exist_ok=False)
    result=read(root/'results.json');audit=read(root/'supplemental_audit.json');frozen=read(root/'frozen_choices.json');trials=read(root/'trials.json')['trials'];plan=read(root/'resolved_configuration.json')
    if audit['status']!='PASS' or audit['results_sha256']!=sha(root/'results.json'):raise ValueError('Require independent full replay audit')
    rows=[];lines=['# Cheap paired-expert routing at approximately half acceptance','',
        '**Approximate reproduction / research extension.** Frozen Text+Kernel ResNet50 Attention U-Net + PARSeq; improved native ViTAEv2-S/TextOCR DeepSolo fallback.',
        '', f"FIT{len(plan['partitions']['fit'])} trains the small heads and shared-context PCA. VALIDATION{len(plan['partitions']['validation'])} chooses among24 candidates using mean hybrid F1 at acceptance quotas{plan['config']['selection_quotas']} and fixes scalar cuts of the unchanged winner. TEST{len(plan['partitions']['test'])} never selects the candidate or thresholds in this run. Historical expert/validation/TEST exposure remains; this is an exploratory study.",
        '', 'Primary metric: per-image lexical-multiset micro F1, Unicode casefold then alphanumeric-only matching, no localization IoU; preserve source native256 care/ignore handling and empty/symbol-only denominators. Actual new DeepSolo calls=0; reported routing counts describe cached output choices.',
        '',f"Selected: `{result['selected']}`; inference dimension{result['deployment_cost']['feature_dimension']}. Shared U-Net pooling required: {result['deployment_cost']['requires_shared_pooling']}. No CLIP/BLIP encoder or fallback result is an inference input.",
        '', '## Identical half-image budget on TEST','',
        'The batch rule accepts the highest-scored150 images, fixed before TEST and without accessing annotations. It is distinct from a scalar threshold transferred from validation. The GT oracle uses privileged annotations and the same frozen experts/eligibility; it is not a deployable method.',
        '', '| Method | F1 (%) | DeepSolo images/total (%) |','| --- | ---: | --- |']
    half=result['splits']['test']['predeclared_batch_quotas']['0.5'];upper=result['splits']['test']['conditional_gt_oracles']['0.5']
    old=result['token_baselines_at_half']['unet'];yolo=result['token_baselines_at_half']['yolo']
    for label,v in [('Previous U-Net token gate',old),('Cheap U-Net gate',half),('U-Net conditional GT oracle',upper),('Previous YOLO token gate',yolo)]:
        lines.append(f"| {label} | {100*v['combined']['f1']:.4f} | {cost(v)} |")
    ds=result['splits']['test']['all_deepsolo'];lines.append(f"| All improved DeepSolo | {100*ds['metrics']['f1']:.4f} | {cost(ds)} |")
    gain=100*(half['combined']['f1']-old['combined']['f1']);regret=100*(upper['combined']['f1']-half['combined']['f1'])
    denominator=upper['combined']['f1']-old['combined']['f1'];closed=(half['combined']['f1']-old['combined']['f1'])/denominator if denominator else None
    interval=result['paired_delta_vs_unet_token_50']
    lines += ['',f"Observed change versus the previous U-Net gate: {gain:+.4f} F1 points, both at{cost(half)} DeepSolo. Conditional RGB-cluster bootstrap95% interval [{interval['lower95_pp']:+.4f},{interval['upper95_pp']:+.4f}] points; training/search variability is not included.",
        '',f"Remaining conditional oracle gap: {regret:.4f} points at{cost(half)} DeepSolo. Fraction of the previous gate-to-oracle gap closed: {100*closed:.2f}%. This is a conditional descriptive ratio, not a generalization guarantee.",
        '', '## Scalar thresholds selected on validation','',
        '| Split | Validation acceptance target | Frozen cutoff | Accepted light images | F1 (%) | DeepSolo images/total (%) |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for split,pop in result['splits'].items():
        for entry in pop['thresholds']:
            policy=entry['policy'];threshold=policy['threshold']
            lines.append(f"| {split.upper()} | {100*policy['validation_target_fraction']:.0f}% | {threshold:.6f} | {entry['accepted_images']}/{entry['total_evaluated_images']} | {100*entry['combined']['f1']:.4f} | {cost(entry)} |")
            rows.append(dict(split=split,rule='validation_scalar',acceptance_target=policy['validation_target_fraction'],threshold=threshold,
                f1=entry['combined']['f1'],accuracy_tn0=entry['combined']['accuracy_tn0'],predicted_words=entry['combined']['detected_words'],
                accepted_images=entry['accepted_images'],deepsolo_images=entry['deepsolo_images'],total_evaluated_images=entry['total_evaluated_images'],deepsolo_percentage=entry['deepsolo_percentage']))
        for quota,entry in pop['predeclared_batch_quotas'].items():
            rows.append(dict(split=split,rule='predeclared_batch_quota',acceptance_target=float(quota),threshold=None,
                f1=entry['combined']['f1'],accuracy_tn0=entry['combined']['accuracy_tn0'],predicted_words=entry['combined']['detected_words'],
                accepted_images=entry['accepted_images'],deepsolo_images=entry['deepsolo_images'],total_evaluated_images=entry['total_evaluated_images'],deepsolo_percentage=entry['deepsolo_percentage']))
    lines += ['', '## Incremental cost','',f"Portable deployment JSON: {result['deployment_cost']['serialized_bytes']} bytes. Model: {result['deployment_cost']['coefficients']} fitted coefficients, {result['deployment_cost']['trees']} trees, {result['deployment_cost']['nodes']} nodes; upper bound{result['deployment_cost']['maximum_comparisons']} tree comparisons per image.",
        '',f"CPU observations + selected projection + head: median{result['deployment_cost']['head_and_observation_median_ms']:.3f}ms, p95{result['deployment_cost']['head_and_observation_p95_ms']:.3f}ms. This excludes U-Net/PARSeq inference and shared pooling; no end-to-end speedup is measured."]
    for split in ('train','test'):
        file=root/(split+'_shared_receipt.json')
        if file.exists():
            r=read(file);peak=r['cuda_peak']
            reuse='Reused qualified cached pooling; actual new U-Net forwards=0.' if 'reused_source_root' in r else 'Offline extraction ran real U-Net forwards; deployment reuses the existing segmentation forward.'
            lines += ['',f"Shared-feature extraction{split}: RGB512, FP32,batch1,{r['device']}; source peak allocated/reserved{peak['allocated']}/{peak['reserved']} bytes. Source added pooling median{r['pooling_median_ms']:.3f}ms,p95{r['pooling_p95_ms']:.3f}ms. Frozen map replay max difference{r['maximum_probability_replay_error']:.8g}; model state unchanged. {reuse}"]
    lines += ['', '## Validation ablations','', '| Features | Training | Best validation mean F1 (%) | DeepSolo images/total (%) at configured acceptance quotas |','| --- | --- | ---: | --- |']
    for feature in ('ocr','layout','shared_context'):
        for aug in (False,True):
            found=max((t for t in trials if t['feature_set']==feature and t['augmented']==aug),key=lambda t:t['selection_objective'])
            costs='; '.join(cost(v) for v in found['validation_points'].values())
            lines.append(f"| {feature} | {'Real + counterfactual' if aug else 'Real only'} | {100*found['selection_objective']:.4f} | {costs} |")
    adaptation=plan['config'].get('expert_unseen_adaptation',{})
    exposure='All router/PCA fitting uses75 expert-unseen images; the1128 expert-fitting labels are excluded.' if adaptation.get('expert_unseen_only') else 'Expert-fitting predictions remain in router training; genuine expert out-of-fold training was not performed.'
    lines += ['', '## Limitations and audit','',
        exposure+' Synthetic candidate omissions do not simulate all mask misses. Small validation and24 candidate choices can overfit selection; validation was previously used for expert checkpoint selection. TEST was historically explored. This does not establish strict paper reproduction or a comparable paper-accuracy improvement.',
        '',f"Independent deterministic full replay{audit['checks']} PASS. It refits all24 heads from actual FIT tensors, recomputes counterfactual lexical labels, verifies FIT-only PCA, validation winner/cut bindings, and replays every TEST single-image decision. The initial auditor's array-layout issue was diagnosed and corrected without changing any model, prediction or comparison tolerance; qualification history is recorded in the experiment guide."]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    with (out/'thresholds.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5.5))
    for scope,label,color in [('predeclared_batch_quotas','Cheap gate: fixed batch quota','tab:blue'),('conditional_gt_oracles','Conditional GT oracle','tab:green')]:
        points=list(result['splits']['test'][scope].values());points.sort(key=lambda p:p['deepsolo_images'])
        ax.plot([p['deepsolo_percentage'] for p in points],[100*p['combined']['f1'] for p in points],marker='o',label=label,color=color)
        for p in points:ax.annotate(f"{p['deepsolo_images']}/{p['total_evaluated_images']} ({p['deepsolo_percentage']:.0f}%)",(p['deepsolo_percentage'],100*p['combined']['f1']),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7)
    for label,p,color in [('Previous U-Net token gate',old,'tab:orange'),('Previous YOLO token gate',yolo,'tab:red')]:
        ax.scatter([p['deepsolo_percentage']],[100*p['combined']['f1']],label=f"{label}: {cost(p)}",color=color,marker='s')
    ax.axhline(100*ds['metrics']['f1'],color='gray',linestyle='--',label=f"All DeepSolo: {cost(ds)}")
    ax.set(xlabel='Images sent to DeepSolo (%)',ylabel='Lexical micro F1 (%)',title='Approximate research: fixed-budget cached-output routing, TEST300')
    ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'routing_curve.png',dpi=180);fig.savefig(out/'routing_curve.svg');plt.close(fig)
    seal(out/'receipt.json',dict(results_sha256=sha(root/'results.json'),supplemental_audit_sha256=sha(root/'supplemental_audit.json'),
        code_sha256=sha(__file__),outputs={f.name:sha(f) for f in out.iterdir() if f.is_file()}))
    print(out/'report.md')


if __name__=='__main__':main()
