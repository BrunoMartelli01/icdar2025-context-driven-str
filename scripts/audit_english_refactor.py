"""Verify English presentation against archived source and pinned original reports.

Only the requested new audit JSON is written. No models, datasets, packages,
training, or inference are loaded. The original source pack is maintainer-held.
"""

import argparse
import ast
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from research.comparison import EvidenceError, compare, digest, load_registry
from research.library import check_saved_views


TRANSLATED_GUIDES = (
    "ADASpot_GAP", "AGENT_EXPERIMENT_REVIEW_2026_10_11", "PAPER_TABLE5_SOURCE_AUDIT",
    "YOLO_CONTEXT_QUALITY_ROUTER", "MASK_CROP_THREE_TESTS", "TOTALTEXT_BERT_CORRECTION",
    "HOST_OPERATIONS", "CLIP_VERIFICATION_RESULTS", "TRAIN_GROUPING_FAILURES_AND_RECOVERY",
    "CLIP_INNOVATIVE_VERIFICATION", "REFACTOR_AUDIT", "GUARDED_WORD_GT_V3", "CLIP_WORD_VERIFICATION",
    "YOLO_RESIDUAL_TEXT_VQA", "CLIP_HIGH_RECALL_ADMISSION", "RESEARCH_SYNTHESIS_20261021",
    "CONTEXTUAL_OCR_CORRECTION_TRAIN_V1", "WORD_CENTERLINE", "README",
)
PROTECTED_PACKAGES = {
    "caption", "cli", "configs", "deepsolo", "docker", "environments", "localize",
    "manifests", "models", "pipeline", "recognition", "scoring", "segmentation",
}
LABEL_SCRIPTS = (
    "scripts/audit_yolo_router_comparison.py", "scripts/clip_router_sweep_visuals.py",
    "scripts/package_text_recall_refinements_review.py",
)


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


class StripStrings(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value="<string>"), node)
        return node


def structure_hash(path):
    tree = StripStrings().visit(ast.parse(path.read_text(encoding="utf-8")))
    return digest(ast.dump(tree, include_attributes=False).encode())


def historical_identities(index):
    result = []
    for entry in index["entries"]:
        identity = {key: value for key, value in entry.items() if key not in ("title", "headings")}
        identity["headings"] = [{key: value for key, value in heading.items() if key != "title"}
                                for heading in entry.get("headings", [])]
        result.append(identity)
    return result


def audit(root, archive, runs_root, artifacts_root):
    baseline = archive / "original-source"
    translations = []
    for name in TRANSLATED_GUIDES:
        path = "docs/" + name + ".md"
        original, current = (baseline / path).read_bytes(), (root / path).read_bytes()
        before, after = (Counter(re.findall(r"\d+", raw.decode())) for raw in (original, current))
        require(not (before - after), "Lost numerical tokens in translated guide: " + path)
        scope = "translation"
        if name == "README":
            scope = "translation plus navigation"
        elif name.startswith("YOLO_"):
            scope = "translation plus concurrent completed-result additions"
        translations.append(dict(path=path, original_sha256=digest(original), english_sha256=digest(current),
                                 numeric_tokens_preserved=before == after, removed_numeric_tokens=dict(before - after),
                                 added_numeric_tokens=dict(after - before), scope=scope))

    protected = []
    for path in sorted(baseline.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(baseline).as_posix()
        if Path(relative).parts[0] not in PROTECTED_PACKAGES:
            continue
        require((root / relative).read_bytes() == path.read_bytes(), "Changed scientific component/input: " + relative)
        protected.append(dict(path=relative, sha256=digest(path.read_bytes())))

    structure = []
    for path in LABEL_SCRIPTS:
        before, after = structure_hash(baseline / path), structure_hash(root / path)
        require(before == after, "Changed Python structure/non-string constants: " + path)
        structure.append(dict(path=path, structure_and_non_string_constants_sha256=before, unchanged=True))

    index_path = "docs/history/WORKSPACE_AUDIT_INDEX.json"
    original_index = json.loads((baseline / index_path).read_text())
    current_index = json.loads((root / index_path).read_text())
    require(historical_identities(original_index) == historical_identities(current_index),
            "Historical evidence identities/positions changed")
    registry = load_registry(root / "research/experiments.json")
    rows = 0
    for group in registry["groups"]:
        original = compare(group, dict(repository=root, runs=runs_root, external_artifacts=artifacts_root))
        saved = check_saved_views(root, group["id"])
        require(original["rows"] == saved["rows"] and original["group"] == saved["group"],
                "Saved summary differs from original verified report: " + group["id"])
        rows += len(saved["rows"])

    prompt_path = "configs/approximate/clip_router_prompt_ablation_v2.json"
    return dict(
        schema_version=1, captured_utc=datetime.now(timezone.utc).isoformat(),
        scope="English presentation and navigation; no new training/inference or scientific result",
        classification="ASSUMPTION",
        original_archive=dict(location=str(archive / "original-source.tar.gz"),
                              sha256=digest((archive / "original-source.tar.gz").read_bytes()),
                              manifest_sha256=digest((archive / "original-manifest.json").read_bytes()),
                              availability="Maintainer-held; no public URL recorded"),
        translations=translations, protected_scientific_files=len(protected),
        protected_path_and_hash_list_sha256=digest(json.dumps(protected, sort_keys=True).encode()),
        protected_prompt_file=prompt_path, protected_prompt_file_sha256=digest((root / prompt_path).read_bytes()),
        python_label_only_changes=structure, historical_index_evidence_identities_unchanged=True,
        historical_index_entries=len(current_index["entries"]), saved_result_groups=len(registry["groups"]),
        saved_result_rows=rows, all_saved_rows_equal_original_verified_comparisons=True,
        concurrent_changes_preserved=["YOLO context/residual VQA completed-result additions",
                                      "Residual VQA test updates", "Ambiguity log status additions"],
        initial_failures_corrected=[
            "CSV newline normalization: raw bytes now compared; Git preserves CRLF",
            "Source export CSV policy narrowed to reviewed result summaries and tested",
            "Remaining historical titles and language-detector fixture corrected",
            "Temporary translation-helper variable/import-path errors corrected before completion",
        ],
        remaining_scientific_blockers="Unchanged; see STRICT_REPRODUCTION_STATUS.md and PAPER_AMBIGUITIES.md",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New audit JSON; existing files are protected")
    args = parser.parse_args(argv)
    try:
        result = audit(args.repo_root.resolve(), args.archive_root.resolve(), args.runs_root, args.artifacts_root)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print("PASS: {} guides, {} unchanged scientific files, {} verified numerical rows".format(
            len(result["translations"]), result["protected_scientific_files"], result["saved_result_rows"]))
        return 0
    except (EvidenceError, OSError, ValueError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
