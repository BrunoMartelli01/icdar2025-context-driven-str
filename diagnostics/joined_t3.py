"""CPU replay of T1/T2/T3 routing with all retained crop texts joined into T3."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from rapidfuzz.fuzz import ratio

from localize.practical_candidate_selection_v2 import form_anchor_lines
from recognition.icdar13_annotations import load_icdar13_recognition_samples
from scoring.semantic import MPNetApproxConfig, MPNetSemanticSimilarity
from scoring.strategies import decide


FAMILIES = ("faithful", "v1_repository", "balanced")
STRATEGIES = ("repository_paired_threshold", "selected_candidate_paired")
CAPTIONS = {"repository": "03-blip2", "antirepetition": "06-practical-captions"}


def join_crop_transcriptions(proposals):
    """Order the existing boxes into lines and join every nonempty crop text.

    Reuses the frozen geometric line-order helper. This never merges or changes
    boxes, filters by confidence, removes repeated words, or consults GT.
    """
    anchors = []
    ids = set()
    for row in proposals:
        identity = str(row["candidate_id"])
        if identity in ids:
            raise ValueError("duplicate candidate ID")
        ids.add(identity)
        box = list(row["bbox"])
        if len(box) != 4 or not all(np.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("invalid crop bbox")
        if not isinstance(row["text"], str):
            raise ValueError("crop text must be a string")
        anchors.append({**row, "candidate_id": identity, "tight_bbox": box})
    lines = form_anchor_lines(anchors)
    ordered = [anchors[i] for line in lines for i in line]
    return {
        "text": " ".join(row["text"].strip() for row in ordered if row["text"].strip()),
        "ordered_blocks": [{"candidate_id": row["candidate_id"], "bbox": row["bbox"],
                            "text": row["text"]} for row in ordered],
        "line_candidate_ids": [[anchors[i]["candidate_id"] for i in line] for line in lines],
        "input_crop_count": len(proposals),
        "nonempty_crop_count": sum(bool(row["text"].strip()) for row in ordered),
        "assembly": "existing geometric line ordering, left-to-right; space join; no deduplication",
    }


def compare_joined(t1, joined, caption, embeddings, strategy):
    """Compare the complete T1 and joined T3 to the same complete T2."""
    if strategy not in STRATEGIES:
        raise ValueError("unsupported joined-T3 strategy")
    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 3 or not np.isfinite(matrix).all():
        raise ValueError("expected finite embeddings for T1, T3, T2")

    def cosine(a, b):
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.clip(np.dot(a, b) / denominator, -1, 1)) if denominator else 0.0

    t3 = joined["text"]
    return decide(
        strategy, T1=t1, T2=caption, T3=t3,
        S1=cosine(matrix[0], matrix[2]), S3=cosine(matrix[1], matrix[2]),
        L1=ratio(t1, caption)/100.0, L3=ratio(t3, caption)/100.0,
    ).to_dict()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_shard(root, stage, sample_id, image_hash, contract, seals):
    name = f"{int(sample_id):04d}.json"
    path = root / "artifacts" / stage / name
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != seals[stage]["sha256"].get(name):
        raise RuntimeError(f"shard SHA mismatch: {path}")
    value = json.loads(raw)
    if (value.get("sample_id") != sample_id or value.get("image_sha256") != image_hash
            or value.get("contract_sha256") != contract):
        raise RuntimeError(f"shard provenance mismatch: {path}")
    return value


def summarize(rows, gt):
    fallback = sum(row["decision"]["needs_fallback"] for row in rows)
    matched = predicted = gt_count = accepted = 0
    before_matched = 0
    for row in rows:
        expected = Counter(word.casefold() for word in gt[row["sample_id"]])
        # Serialized output words, including all joined T3 blocks when selected.
        population = Counter(row["final_text"].casefold().split())
        before = Counter(row["decision"]["selected_output"].casefold().split())
        matched += sum((population & expected).values())
        before_matched += sum((before & expected).values())
        predicted += sum(population.values())
        gt_count += sum(expected.values())
        accepted += not row["decision"]["needs_fallback"] and bool(population & expected)
    return {
        "images": len(rows), "gt_words": gt_count, "fallback_images": fallback,
        "fallback_rate": fallback/len(rows), "accepted_images": len(rows)-fallback,
        "accepted_images_with_at_least_one_correct_word": accepted,
        "selected_T1_images": sum(r["decision"]["semantic_winner"] == "T1" for r in rows),
        "selected_T3_images": sum(r["decision"]["semantic_winner"] == "T3" for r in rows),
        "pre_fallback_word_matches": before_matched,
        "final_word_matches": matched, "final_predicted_words": predicted,
        "word_precision": matched/predicted if predicted else 0.0,
        "word_recall": matched/gt_count,
        "word_f1": 2*matched/(predicted+gt_count),
        "confidence_median": float(np.median([r["decision"]["C"] for r in rows])),
        "confidence_max": max(r["decision"]["C"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("suite-root", "hf-cache", "data-root", "recognition-manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_CACHE"] = str(args.hf_cache.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    root = args.suite_root.resolve()
    source_path = root / "results.json"
    source = json.loads(source_path.read_text())
    receipt = json.loads((root / "artifacts/report-receipt.json").read_text())
    source_hash = sha(source_path)
    contract = source["contract_sha256"]
    if receipt != {"contract_sha256": contract, "report_sha256": source_hash}:
        raise RuntimeError("suite receipt mismatch")
    if source["scope"] != {"split": "TEST", "images": 233, "gt_words": 1095}:
        raise RuntimeError("requires frozen TEST-233 suite")
    if sha(args.recognition_manifest) != source["protocol"]["inputs"]["recognition_manifest"]:
        raise RuntimeError("recognition manifest SHA mismatch")
    stages = ("07-suite-routing", "05-deepsolo", *CAPTIONS.values())
    seals = {stage: json.loads((root / "artifacts/seals" / f"{stage}.json").read_text()) for stage in stages}
    for seal in seals.values():
        if seal["contract_sha256"] != contract or len(seal["sha256"]) != 233:
            raise RuntimeError("invalid source stage seal")
    inputs = []
    for i in range(1, 234):
        sid = str(i)
        # The routing shard is anchored to the source seal before its image hash is used.
        path = root / "artifacts/07-suite-routing" / f"{i:04d}.json"
        if sha(path) != seals["07-suite-routing"]["sha256"][path.name]:
            raise RuntimeError("routing checksum mismatch")
        image_hash = json.loads(path.read_text())["image_sha256"]
        values = {stage: checked_shard(root, stage, sid, image_hash, contract, seals) for stage in stages}
        routing = values["07-suite-routing"]
        deep = values["05-deepsolo"]["response"]
        if deep.get("success") is not True:
            raise RuntimeError("failed cached DeepSolo result")
        inputs.append({"sample_id": sid, "image_sha256": image_hash,
                       "T1": routing["faithful"]["route"]["T1"],
                       "joined": {family: join_crop_transcriptions(routing[family]["proposals"]) for family in FAMILIES},
                       "captions": {profile: values[stage]["caption"]["raw_caption"] for profile, stage in CAPTIONS.items()},
                       "fallback_text": deep["selected_output"]})
    results = {f"{family}/{profile}/{strategy}": [] for family in FAMILIES for profile in CAPTIONS for strategy in STRATEGIES}
    scorer = MPNetSemanticSimilarity(MPNetApproxConfig(device="cpu", precision="float32"))
    runtime = scorer.metadata
    try:
        for number, sample in enumerate(inputs, 1):
            texts = [sample["T1"]] + [sample["joined"][f]["text"] for f in FAMILIES] + list(sample["captions"].values())
            embeddings = scorer.embeddings(texts)
            for fi, family in enumerate(FAMILIES, 1):
                for pi, profile in enumerate(CAPTIONS, 4):
                    for strategy in STRATEGIES:
                        decision = compare_joined(sample["T1"], sample["joined"][family], sample["captions"][profile],
                                                  embeddings[[0, fi, pi]], strategy)
                        results[f"{family}/{profile}/{strategy}"].append({
                            "sample_id": sample["sample_id"], "decision": decision,
                            "final_text": sample["fallback_text"] if decision["needs_fallback"] else decision["selected_output"],
                        })
            print(f"[T3 joined / CPU] {number}/233 sample={sample['sample_id']}", flush=True)
    finally:
        scorer.release()
    # GT is first opened after the complete routing result is fixed.
    samples, _ = load_icdar13_recognition_samples(args.recognition_manifest, args.data_root)
    if len(samples) != 233 or sum(len(s.words) for s in samples) != 1095:
        raise RuntimeError("GT cardinality mismatch")
    expected_images = {r["sample_id"]: r["image_sha256"] for r in inputs}
    if any(expected_images[s.sample_id] != s.image_sha256 for s in samples):
        raise RuntimeError("GT/source image identity mismatch")
    gt = {s.sample_id: [w.transcription for w in s.words] for s in samples}
    summaries = {name: summarize(rows, gt) for name, rows in results.items()}
    report = {
        "mode": "t3-joined-replay-v1", "reproduction_mode": "approximate",
        "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC", "table_5_accuracy": None,
        "scope": source["scope"], "device": "cpu", "new_segmentation_or_ocr_or_caption_inference": False,
        "source_report_sha256": source_hash, "source_contract_sha256": contract,
        "source_stage_seal_sha256": {stage: sha(root/"artifacts/seals"/f"{stage}.json") for stage in stages},
        "mpnet": runtime,
        "cpu_thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "implementation_sha256": sha(Path(__file__)),
        "ordering_helper_sha256": sha(Path(__file__).resolve().parents[1]/"localize/practical_candidate_selection_v2.py"),
        "alpha": .6, "beta": .4, "tau": .8,
        "fallback_reduction": "highest_score, unchanged from source experiment",
        "metric": "whitespace tokens of serialized final text; per-image casefold one-to-one multiset; no spatial matching",
        "scientific_choices": {"T3_join": "ASSUMPTION: user-requested concatenation of every retained block",
                               "reading_order": "ASSUMPTION: reuse frozen line geometry on retained boxes",
                               "repository_strategy": "REPOSITORY_SPECIFIED: maximum paired C, semantic-winner text",
                               "selected_strategy": "ASSUMPTION: paper-prose-aligned paired C of semantic winner"},
        "assembled_inputs": inputs, "summaries": summaries, "per_image": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temporary, args.output)
    finally:
        temporary.unlink()
    print(json.dumps(summaries, indent=2))
    print(f"JSON {args.output}\nSHA256 {sha(args.output)}", flush=True)


if __name__ == "__main__":
    main()
