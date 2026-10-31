"""Export audited matched-loss curves, counts and historical emission controls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

from scripts.audit_icdar13_matched_word_training import read, digest


def selection_geometry(root, plan):
    """Post-hoc native geometry at the fixed policy; never selects a model."""
    import numpy as np
    import torch
    from segmentation.icdar13_word_refit import WordDataset
    from segmentation.totaltext_word_kernel import instance_geometry
    from segmentation.word_multitask_evaluation import aggregate_word_metrics
    from recognition.unet_token_routing import unet_proposals
    torch.set_num_threads(plan["config"]["cpu_threads"])
    lookup = {r["sample_id"]: r for r in plan["train_records"]}
    rs = [lookup[s] for s in plan["partitions"]["selection"]]
    dataset = WordDataset(rs, plan["config"], augment=False)
    targets = [dataset[i] for i in range(len(rs))]
    result = []
    for name in ["source"] + plan["config"]["branches"]:
        totals, rows = {}, []
        for r, target in zip(rs, targets):
            path = root / "policy_selection" / name / "maps" / (r["sample_id"].replace(":", "__") + ".npz")
            with np.load(path) as archive:
                _, labels, _ = unet_proposals(archive["probabilities"], r["image_wh"], plan["config"]["inference"])
            valid = target["validity"][0].numpy().astype(bool)
            care = labels.copy()
            for label in np.unique(care[care > 0]):
                if not np.any((care == label) & valid):
                    care[care == label] = 0
            metrics = instance_geometry(care, target["owners"].numpy(), list(range(1, r["care_words"]+1)))
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0) + v
            rows.append(dict(sample_id=r["sample_id"], maps_sha256=digest(path), **metrics))
        result.append(dict(candidate=name, aggregate=aggregate_word_metrics(totals), per_image=rows,
            deepsolo_images=0, total_images=len(rs), deepsolo_percentage=0.))
    return dict(mode="approximate", claim_label="approximate reproduction", split="selection", model_selection_usage=False,
        protocol="All native instances before cap/OCR; fixed text/kernel thresholds; fully-invalid exclusion; maximum-cardinality native bbox IoU>=.5; any-pixel split/fusion proxies", results=result)


def render_plot(root):
    """Render saved numbers only; this interpreter does not run any model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = root / "report"
    result, plan = read(root / "results.json"), read(root / "resolved_configuration.json")
    with (out / "comparison.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    source_f1 = read(root / "source/selection/metrics.json")["counts"]["f1"]*100
    for branch in plan["config"]["branches"]:
        h = read(root / branch / "epochs.json")
        points = [(0, source_f1)] + [(r["epoch"], r["selection"]["f1"]*100) for r in h if "selection" in r]
        axes[0].plot([x for x, _ in points], [y for _, y in points], marker="o", label=branch)
    axes[0].set(xlabel="Training epoch", ylabel="Selection lexical micro F1 (%)", title="TRAIN selection: fixed OCR policy")
    axes[0].legend(); axes[0].grid(alpha=.2)
    test = [r for r in rows if r["split"] == "test"]
    selected_label = {"source": "source", "refit_bce": "BCE", "totaltext_bce_dice": "BCE+Dice"}[result["winner"]["candidate"]]
    labels = ["Native\nhistorical", "Corrected\nTotal-Text", "Source\nmatched fixed", "Selected\n" + selected_label]
    axes[1].bar(labels, [float(r["f1_percent"]) for r in test], color=["#b0b4bb", "#7a89a6", "#d2a45b", "#277e8e"])
    for i, r in enumerate(test):
        value = float(r["f1_percent"])
        axes[1].text(i, value+.3, f'{value:.2f}', ha="center")
    n, words = len(plan["partitions"]["test"]), plan["config"]["expected_test_care_words"]
    axes[1].set(ylabel="TEST lexical micro F1 (%)", title=f"{n} images /{words} care words; DeepSolo0/{n} (0.00%)", ylim=(0, 100))
    fig.suptitle("ICDAR2013 — approximate reproduction; no IoU matching")
    fig.tight_layout(); fig.savefig(out / "training_and_test.png", dpi=170); plt.close(fig)


def report(root, native_baseline, totaltext_baseline, plot_python):
    result, plan = read(root / "results.json"), read(root / "resolved_configuration.json")
    audit = read(root / "independent_lexical_audit.json")
    if audit["status"] != "PASS" or audit["results_sha256"] != digest(root / "results.json") or plan["smoke"]:
        raise ValueError("Complete audited approximate experiment required")
    baselines = [("Historical native refit, common emission", read(native_baseline)),
                 ("Historical corrected Total-Text, common emission", read(totaltext_baseline))]
    out = root / "report"
    out.mkdir(exist_ok=False)
    rows = []
    for name, baseline in baselines:
        n = plan["config"]["expected_populations"]["test"]
        if baseline["test_images"] != n or baseline["total_images"] != n or baseline["counts"]["tp"] + baseline["counts"]["fn"] != plan["config"]["expected_test_care_words"]:
            raise ValueError("Historical denominator differs")
        rows.append(dict(name=name, split="test", f1_percent=baseline["f1"]*100, **baseline["counts"],
            deepsolo_images=0, total_images=n, deepsolo_percentage=0.))
    for value in result["results"]:
        counts = value["counts"]
        rows.append(dict(name=value["candidate"], split=value["split"], f1_percent=counts["f1"]*100,
            **{k: counts[k] for k in ("tp", "fp", "fn")}, deepsolo_images=0,
            total_images=value["total_images"], deepsolo_percentage=0.))
    with (out / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    geometry = selection_geometry(root, plan)
    (out / "selection_geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
    # Preserve the virtualenv executable path rather than following its symlink
    # to the bare interpreter, which would select different site-packages.
    subprocess.run([str(plot_python.absolute()), "-m", "scripts.report_icdar13_matched_word_training",
        "--run-root", str(root), "--plot-only"], cwd=Path(__file__).resolve().parents[1], check=True)
    winner = result["winner"]
    deployment = dict(mode="approximate", claim_label="approximate reproduction", expert="U-Net Text+Kernel / PARSeq",
        checkpoint=winner["checkpoint"], checkpoint_sha256=winner["checkpoint_sha256"],
        architecture="WordRefitUNet(partial_encoder=False); every state key required",
        preprocessing={k: plan["config"][k] for k in ("image_hw", "canvas_fill_rgb", "probability_threshold", "skip_floor", "initial_gap_bias")},
        policy=winner["settings"], reject_no_alphanumeric_emissions=True,
        parseq_paths={k: plan["paths"][k] for k in ("parseq_source", "parseq_archive", "parseq_checkpoint")},
        source_plan=str(root / "resolved_configuration.json"), source_plan_sha256=digest(root / "resolved_configuration.json"),
        selection_source=str(root / "selection_frozen.json"), selection_sha256=digest(root / "selection_frozen.json"),
        deepsolo_images=0, total_images=plan["config"]["expected_populations"]["test"], deepsolo_percentage=0.)
    (out / "selected_expert.json").write_text(json.dumps(deployment, indent=2) + "\n")
    lines = ["# ICDAR2013 matched word-loss experiment", "", "Approximate reproduction. Independent CSV/Counter audit: PASS.", "",
        f'Selected candidate: `{winner["candidate"]}`; checkpoint `{winner["checkpoint_sha256"]}`.', "",
        "Selected policy: `"+json.dumps(winner["settings"], sort_keys=True)+"`.", "",
        "| Expert | Population | TP | FP | FN | F1 % | DeepSolo images/total (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        lines.append(f'| {r["name"]} | {r["split"]} | {r["tp"]} | {r["fp"]} | {r["fn"]} | {r["f1_percent"]:.4f} | 0/{r["total_images"]} (0.00%) |')
    lines += ["", "Per-image lexical multiset micro F1: Unicode casefold+isalnum, inherited native256 ignore, no IoU. All care TEST words remain in the denominator. All new predictions and confidence policies were selected on TRAIN selection images before CAL/TEST scoring. No new DeepSolo calls or routing.", ""]
    for label, baseline in baselines:
        removed = sum(len(r["rejected"]) for r in baseline["changed_images"])
        lines += [f'{label}: original F1 {100*baseline["original_f1"]:.4f}%; rejecting {removed} symbol-only outputs gives {100*baseline["f1"]:.4f}%, with the same {baseline["counts"]["tp"]} correct words.', ""]
    lines += ["Common emission controls isolate this denominator effect from new training. Historical corrected-Total-Text additionally includes graph, verifier and editor modules; it is a pipeline reference, not a matched training-dataset ablation.", "",
        f'The new loss arms share all other training settings and source state. The pilot has{plan["config"]["training"]["epochs"]} epochs per arm. Small reused selection sets and historical TEST exposure limit generalization claims. This is not official polygon Hmean or paper Table5 accuracy.', "",
        "Post-hoc TRAIN selection geometry uses all native instances before cap/OCR, maximum-cardinality bbox IoU>=.5 matching and any-pixel split/fusion proxies. This different metric does not select a model.", "",
        "| Candidate | Native proposal F1 % | Words touched by fused instances | Words split across instances | DeepSolo images/total (%) |",
        "| --- | ---: | ---: | ---: | --- |"]
    for item in geometry["results"]:
        v = item["aggregate"]
        lines.append(f'| {item["candidate"]} | {100*v["proposal_f1_iou_50"]:.4f} | {v["gt_words_touched_by_fused_components"]} | {v["gt_words_split_across_predicted_components"]} | 0/{item["total_images"]} (0.00%) |')
    lines += ["", "![Training and test comparison](training_and_test.png)", ""]
    (out / "SUMMARY.md").write_text("\n".join(lines))
    inputs = [root / "results.json", root / "resolved_configuration.json", root / "independent_lexical_audit.json", native_baseline, totaltext_baseline]
    metadata = dict(mode="approximate", claim_label="approximate reproduction", inputs_sha256={str(p): digest(p) for p in inputs},
        outputs_sha256={p.name: digest(p) for p in out.iterdir() if p.is_file()}, producer_sha256=digest(__file__),
        plot_interpreter=str(plot_python.absolute()), plotting_scope="Saved numbers only; neural/target computations stay in the qualified main environment")
    (out / "report_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--native-baseline", type=Path)
    parser.add_argument("--totaltext-baseline", type=Path)
    parser.add_argument("--plot-python", type=Path)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        render_plot(args.run_root)
    else:
        if not all((args.native_baseline, args.totaltext_baseline, args.plot_python)):
            parser.error("Explicit baseline files and plotting interpreter required")
        print(json.dumps(report(args.run_root, args.native_baseline, args.totaltext_baseline, args.plot_python)))


if __name__ == "__main__":
    main()
