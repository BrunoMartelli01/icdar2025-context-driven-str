"""Readable before/after tables and standalone scientific routing curves."""

import argparse
import csv
from pathlib import Path

from recognition.router_fallback_replay import ordered_frontier
from scripts.icdar_budget_data import read, sha


def report(c, document, plot_only=False):
    root=Path(c["study_root"])
    result=read(root/"results.json")
    audit=read(root/"independent_audit.json")
    assert audit["status"] == "PASS" and audit["results_sha256"] == sha(root/"results.json")
    lines=["# Frozen trained routers with ICDAR-specific DeepSolo vocabularies", "",
        "**Approximate reproduction / research extension.** Full official TEST: ICDAR2013 233 images /1095 care words; ICDAR2015 500 images /2077 care words. Logical image routing on cached expert outputs; zero new neural inferences and no training in this replay.", "",
        "Use the existing dataset-specific trained gates from `ICDAR_TRAINED_BUDGET_ROUTER.md`, not the earlier TotalText-only transferred essential head. Keep every selected weight, light-expert output, eligibility guard, RGB tie key, validation scalar cutoff and original batch quota unchanged. The ICDAR2013 U-Net is the selected TotalText corrected chain; ICDAR2015 U-Net is the selected native `long768` chain. YOLO uses its original direct640/PARSeq expert and OCR-only small boosted head. Only the fallback expert changes.", "",
        "ICDAR2013 fallback: ViTAEv2-S+TextOCR TotalText checkpoint, RGB1024/max1892, Generic vocabulary, neural score0.40 / character0.0, keep-unmatched; this is the prior TRAIN-calibration-selected profile. ICDAR2015 fallback: ICDAR2015-finetuned ViTAEv2-S+TextOCR checkpoint, RGB1440/max4000, Weak vocabulary, score0.35 / character0.85, original edit-distance<1.5 rejection. Weak is released benchmark side information; this is not a no-vocabulary result. No lexicon is applied to U-Net/PARSeq or YOLO/PARSeq.", "",
        "## Fixed predeclared budgets", "",
        "Ranks use only unchanged model scores and RGB hashes. Batch-dependent decision boundaries are not validation-fixed scalar thresholds. All rows retain complete care GT and unchanged lexical microF1 normalization (Unicode casefold then alphanumeric characters), per-image multiset matching, no length filtering, no localization IoU and unmatched empty-normalized denominators.", "",
        "| Dataset | Light expert | Previous F1 % | New F1 % | Gain pp | DeepSolo images/total (%) |",
        "| --- | --- | ---: | ---: | ---: | --- |"]
    for ds,arms in result["datasets"].items():
        for arm,v in arms.items():
            p=v["policies"]["batch_"+str(v["primary_acceptance"])]
            lines.append(f"| {ds} | {arm} ({v['expert']}) | {100*p['previous_f1']:.4f} | {100*p['f1']:.4f} | {p['gain_over_previous_pp']:.4f} | {p['deepsolo_images']}/{p['total_images']} ({p['deepsolo_percent']:.2f}%) |")
    lines += ["", "## Original validation scalar cuts", "",
        "No cut is recalibrated using the new fallback or TEST labels. The source calibration populations remain23 IC13 and50 IC15 images, separate from source head selection. Actual TEST fallback fractions differ from calibration budgets.", "",
        "| Dataset | Expert | Target light acceptance % | Frozen cutoff | Previous F1 % | New F1 % | DeepSolo images/total (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for ds,arms in result["datasets"].items():
        for arm,v in arms.items():
            for name,p in v["policies"].items():
                if name.startswith("validation_"):
                    lines.append(f"| {ds} | {arm} | {100*p['policy']['target_acceptance']:.0f} | {p['policy']['threshold']:.10f} | {100*p['previous_f1']:.4f} | {100*p['f1']:.4f} | {p['deepsolo_images']}/{p['total_images']} ({p['deepsolo_percent']:.2f}%) |")
    lines += ["", "## Full DeepSolo endpoint and complete rank curves", "",
        "The original trained-router ignore implementation is kept for both before/after experts. IC13 has no changed ignore decision. IC15 differs from the previous standalone lexicon report on five predicted polygons: this router rebuilds native256 validity from clipped annotations and lets care polygons override ignored regions; the standalone study used the prepared validity PNG. A five-image diagnostic confirms that care-region override, rather than coordinate clipping, explains all five changed decisions (`ignore_policy_diagnostic.json`). All2077 care words and500 images are identical. Exactly five outputs move from ignored to care and add five FP without new TP: TP1614, FP208, FN463 gives82.7905%, DeepSolo500/500 (100.00%), versus the prior standalone82.8968%, DeepSolo500/500 (100.00%). This protocol detail is explicit rather than silently mixing denominators. The standalone lexicon result is preserved.", "",
        "| Dataset | DeepSolo baseline | TP | FP | FN | F1 % | DeepSolo images/total (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for ds,arms in result["datasets"].items():
        p=next(iter(arms.values()))["policies"]["all_deepsolo"]
        lines.append(f"| {ds} | {c['new_fallback'][ds]['label']} | {p['tp']} | {p['fp']} | {p['fn']} | {100*p['f1']:.4f} | {p['deepsolo_images']}/{p['total_images']} ({p['deepsolo_percent']:.2f}%) |")
    lines += ["", "Every prefix of the frozen score ordering is also evaluated. Equal scores can be split only by the original RGB tie key. These TEST-descriptive rank extrema are not deployable validation choices and are not GT-oracle upper bounds.", "",
        "| Dataset | Expert | Descriptive point | F1 % | DeepSolo images/total (%) |",
        "| --- | --- | --- | ---: | --- |"]
    for ds,arms in result["datasets"].items():
        for arm,v in arms.items():
            for kind in ("test_descriptive_best", "test_descriptive_minimum_deepsolo_no_loss"):
                p=v[kind]
                lines.append(f"| {ds} | {arm} | {kind} | {100*p['f1']:.4f} | {p['deepsolo_images']}/{p['total_images']} ({p['deepsolo_percent']:.2f}%) |")
    lines += ["", "## Interpretation and uncertainty", "",
        "The stronger fallback improves all four original primary-budget hybrids, with YOLO remaining stronger than the selected U-Net. The unchanged heads were supervised against the old weaker fallback, so their utility targets are mismatched to the new experts. They do not maintain new DeepSolo-only performance at the original budgets. The complete frozen ordering has no hybrid gain over the new IC15 fallback; IC13 gains at its descriptive extrema are small, historically TEST-exposed, and require most images to run DeepSolo. No TEST-selected threshold is promoted.", "",
        "| Dataset | Expert | Rule | F1 95% interval | Gain over new DeepSolo 95% interval pp | DeepSolo images/total (%) |",
        "| --- | --- | --- | --- | --- | --- |"]
    for ds,arms in result["datasets"].items():
        for arm,v in arms.items():
            for name in ("batch_"+str(v["primary_acceptance"]),"validation_"+str(v["primary_acceptance"])):
                p=v["policies"][name];u=p["uncertainty"];a,b=u["f1_interval95_percent"];x,y=u["gain_over_new_deepsolo_interval95_pp"]
                lines.append(f"| {ds} | {arm} | {name} | [{a:.3f}, {b:.3f}] | [{x:.3f}, {y:.3f}] | {p['deepsolo_images']}/{p['total_images']} ({p['deepsolo_percent']:.2f}%) |")
    lines += ["", "Intervals use2000 paired image-bootstrap draws, seed20261028, conditional on fixed experts/heads/decisions; fitting, selection, pretraining overlap and historical TEST exposure are omitted. These word lexical scores are not official polygon Hmean or verified equivalence with paper Table5 accuracy.", "",
        "## Verification and evidence", "",
        f"External evidence: `{root}`. `thresholds.csv` contains every original frozen cut/quota and before/after counts; `frontiers/` contains every ordered prefix; `per_image/` binds branches, unchanged scores and model hashes; `fallback_evaluation/` preserves corrected strings, geometry, GT and counts; `prediction_only/` contains733 GT-free fallback exports; `input_evidence.json` preserves raw hashes and all changed ignore decisions.", "",
        "The independent audit recounts all fallback/light predictions, replays every deployed single-image gate and frozen decision, verifies maps/weights and every frontier point. The initial auditor incorrectly expected the DeepSolo envelope's `payload_sha256` field in plain lightweight-expert JSON and stopped with KeyError before producing an audit. Its executable and traceback are preserved in `failures/audit_schema_v1/`; the corrected auditor uses the actual original cache schema and independently reproduces scores/counts. The first plot attempt found no matplotlib in the main environment after writing its draft document. No package was installed: final plotting uses the existing YOLO environment, with the failed trace and draft document preserved under `failures/plot_environment_v1/`. No scientific predictions or reported metric changed during these repairs.", "",
        "Source classifications: PAPER_SPECIFIED Section3.5/Algorithm2 motivation only; REPOSITORY_SPECIFIED existing qualified gate/expert/metric procedures; ASSUMPTION user-requested frozen-fallback substitution and harmonized original ignore protocol; HARDWARE_DEVIATION CPU cached replay, with historical isolated RTX3090 expert runs; UNRESOLVED metric equivalence, small validation, historical exposure, target mismatch and online latency. STRICT configurations are unchanged.", "",
        "Config: `configs/approximate/trained_router_vocabulary_replay_v1.json`. Implementation: `recognition/router_fallback_replay.py`, `scripts/replay_trained_router_vocabularies.py`, `scripts/audit_trained_router_vocabulary_replay.py`, `scripts/report_trained_router_vocabulary_replay.py`. Tests: `tests/recognition/test_router_fallback_replay.py` plus the relevant original gate/deployment/lexicon/metric/frontier suites.", "",
        f"[Standalone PNG]({root}/figures/frontiers.png), [SVG]({root}/figures/frontiers.svg). Every plotted curve has explicit DeepSolo count axes, full denominators, percentage axes and machine-readable per-point counts."]
    if not plot_only:
        with Path(document).open("x") as f:f.write("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    dense=[]
    for col,(ds,arms) in enumerate(result["datasets"].items()):
        for row,(arm,v) in enumerate(arms.items()):
            ax=axes[row,col];data=read(root/"per_image"/(ds+"__"+arm+".json"))
            curves=dict(previous=ordered_frontier(data["previous_branches"],data["scores"],data["tie_keys"]),
                        updated=read(root/"frontiers"/(ds+"__"+arm+".json"))["points"])
            n=len(data["scores"])
            for name,curve in curves.items():
                color="#999999" if name=="previous" else "#1769aa"
                ax.plot([p["deepsolo_images"] for p in curve],[100*p["f1"] for p in curve],color=color,label=name)
                for p in curve:dense.append(dict(dataset=ds,light_expert=arm,version=name,**p))
            endpoint=v["policies"]["all_deepsolo"]
            ax.axhline(100*endpoint["f1"],ls=":",color="#1769aa",label=f"Updated DeepSolo only: {n}/{n} (100%)")
            for name,color,offset in (("batch_","#db5500",-35),("validation_","#258040",20)):
                p=v["policies"][name+str(v["primary_acceptance"])];x=p["deepsolo_images"];y=100*p["f1"]
                ax.scatter([x],[y],color=color,zorder=5)
                label=f"{'Budget' if name=='batch_' else 'VAL cut'}: F1 {y:.2f}%\n{x}/{n} ({p['deepsolo_percent']:.2f}%)"
                ax.annotate(label,(x,y),xytext=(-40,offset),textcoords="offset points",fontsize=8,color=color,
                            bbox=dict(facecolor="white",edgecolor="none",alpha=.8))
            ax.set_title(ds.upper()+" / "+arm.upper());ax.set_xlabel(f"Images sent to DeepSolo / {n}");ax.set_ylabel("Word lexical micro F1 (%)")
            ax.set_xlim(0,n*1.03);ax.grid(alpha=.2);ax.legend(fontsize=7,loc="lower right")
            sec=ax.secondary_xaxis("top",functions=(lambda x,n=n:100*x/n,lambda x,n=n:x*n/100));sec.set_xlabel("Images sent to DeepSolo (%)")
    fig.suptitle("Approximate reproduction — cached routing, unchanged trained gates\nOriginal budgets and validation cuts; complete TEST populations",fontsize=12)
    fig.tight_layout(rect=[0,0,1,.94]);(root/"figures").mkdir(exist_ok=True)
    for ext in ("png","svg"):fig.savefig(root/"figures"/("frontiers."+ext),dpi=150)
    plt.close(fig)
    with (root/"dense_frontiers.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(dense[0]));w.writeheader();w.writerows(dense)


if __name__ == "__main__":
    p=argparse.ArgumentParser(__doc__);p.add_argument("--config",type=Path,required=True);p.add_argument("--document",type=Path,required=True);p.add_argument("--plot-only",action="store_true");args=p.parse_args()
    report(read(args.config),args.document,args.plot_only)
