"""Export calibrated versus TEST-diagnostic F1/cost tables and scientific figures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def threshold_text(cut):
    return "accept none" if cut["kind"] == "accept_none" else format(cut["value"], ".17g")


def routing_text(point):
    return f'{point["deepsolo_images"]}/{point["total_images"]} ({point["deepsolo_percentage"]:.2f}%)'


def report(root, out):
    result = read(root / "results.json")
    audit = read(root / "independent_audit.json")
    if audit["status"] != "PASS" or audit["results_sha256"] != sha(root / "results.json"):
        raise ValueError("Independent recount qualification required")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for case, value in result["results"].items():
        for label, point in (("all_deepsolo", value["all_deepsolo"]),
                             ("all_light", value["all_light"]),
                             ("calibration_frozen_test", value["primary"]),
                             ("test_label_diagnostic_only", value["test_cut_diagnostic"]["winner"])):
            threshold = value["threshold"] if label == "calibration_frozen_test" else (
                value["test_cut_diagnostic"]["winner"]["threshold"] if label == "test_label_diagnostic_only" else None)
            rows.append(dict(case=case, result_class="approximate reproduction", evaluation=label,
                             threshold=threshold_text(threshold) if threshold else "not applicable",
                             f1_percent=100*point["combined"]["f1"], light_images=point["light_images"],
                             deepsolo_images=point["deepsolo_images"], total_images=point["total_images"],
                             deepsolo_percentage=point["deepsolo_percentage"],
                             tp=point["combined"]["tp"], fp=point["combined"]["fp"], fn=point["combined"]["fn"]))
    with (out / "comparison.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Common-pipeline DeepSolo-parity threshold study", "",
             "Approximate reproduction; complete TEST, frozen CAL-selected thresholds, cached logical image routing.", "",
             "DeepSolo-only references: Total-Text 82.6540%, 300/300 (100.00%); ICDAR2015 82.7905%, 500/500 (100.00%).", "",
             "## Thresholds selected only on reserved CAL, applied unchanged to TEST", "",
             "| Case | Threshold (score >=) | Light images | DeepSolo images/total (%) | TEST F1 % | Difference, points |", "| --- | ---: | ---: | --- | ---: | ---: |"]
    for case, value in result["results"].items():
        p = value["primary"]
        lines.append(f'| {case} | {threshold_text(value["threshold"])} | {p["light_images"]} | {routing_text(p)} | {100*p["combined"]["f1"]:.4f} | {value["delta_f1_points"]:+.4f} |')
    lines += ["", "## Maximum acceptance from TEST labels: diagnostic only", "",
              "These thresholds exhaust the same fixed score ordering using TEST labels. They cannot be promoted as validation-selected deployment policies.", "",
              "| Case | TEST-derived threshold | Light images | DeepSolo images/total (%) | TEST F1 % |", "| --- | ---: | ---: | --- | ---: |"]
    for case, value in result["results"].items():
        p = value["test_cut_diagnostic"]["winner"]
        lines.append(f'| {case} | {threshold_text(p["threshold"])} | {p["light_images"]} | {routing_text(p)} | {100*p["combined"]["f1"]:.4f} |')
    lines += ["", "No useful nontrivial parity policy is validated on both datasets by this frozen common gate.",
              "The TEST-derived maxima characterize scalar cuts of these scores, not arbitrary image subsets or new models.",
              "Metric: per-image Unicode casefold/isalnum exact multiset match, micro TP/FP/FN, no spatial match; inherited ignore/care handling, all images retained.",
              "Historical TEST exploration and upstream expert/vocabulary dependencies prevent an untouched confirmation claim.",
              f'Independent audit: {audit["per_image_recounts"]} per-image recounts and {audit["scalar_cut_replays"]} scalar-cut replays, PASS.', ""]
    (out / "report.md").write_text("\n".join(lines))
    plot_runtime = plot(root, out, result)
    with (out / "report_manifest.json").open("x") as stream:
        json.dump(dict(source_root=str(root), results_sha256=sha(root / "results.json"),
                       audit_sha256=sha(root / "independent_audit.json"), reporter_sha256=sha(__file__),
                       plot_runtime=plot_runtime,
                       output_sha256={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}), stream, indent=2)
        stream.write("\n")


def plot(root, out, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["totaltext__unet", "totaltext__yolo", "icdar2015__unet", "icdar2015__yolo"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    for ax, case in zip(axes.flat, order):
        value = result["results"][case]
        curve = read(root / "test" / (case + "__test_cut_diagnostic.json"))["trials"]
        n = value["primary"]["total_images"]
        ds = value["all_deepsolo"]["combined"]["f1"]
        ax.plot([100*p["light_images"]/n for p in curve], [100*p["combined"]["f1"] for p in curve],
                color="#31688e", label="TEST scalar-cut curve (diagnostic)")
        ax.axhline(100*ds, color="#555555", ls="--", label=f"DeepSolo: {n}/{n} (100%)")
        p = value["primary"]
        ax.scatter(100*p["light_images"]/n, 100*p["combined"]["f1"], marker="s", s=65, color="#d1495b", zorder=5,
                   label="CAL-frozen: " + routing_text(p))
        diag = value["test_cut_diagnostic"]["winner"]
        ax.scatter(100*diag["light_images"]/n, 100*diag["combined"]["f1"], marker="*", s=110, color="#238b45", zorder=6,
                   label="TEST-label best: " + routing_text(diag))
        ax.set(xlabel="Light-expert images (%)", ylabel="Lexical micro F1 (%)", xlim=(-1, 101),
               title=case.replace("__", " / "))
        ax.grid(alpha=.2)
        ax.legend(loc="lower left", fontsize=8)
        top = ax.secondary_xaxis("top")
        ticks = [0, 25, 50, 75, 100]
        top.set_xticks(ticks, [f"{round(n*(1-t/100))}/{n}\n({100-t}%)" for t in ticks])
        top.set_xlabel("DeepSolo images/total (%)")
    fig.suptitle("Approximate reproduction: frozen common router / F1 versus logical fallback cost\n"
                 "Whole score ties; TEST curve uses labels only for diagnosis; zero new neural inference", fontsize=13)
    fig.savefig(out / "f1_fallback_frontier.png", dpi=180)
    fig.savefig(out / "f1_fallback_frontier.pdf")
    plt.close(fig)
    return dict(python=sys.version, executable=sys.executable, matplotlib=matplotlib.__version__)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report(args.run_root, args.output_dir)


if __name__ == "__main__":
    main()
