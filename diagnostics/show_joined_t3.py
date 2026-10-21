"""Read-only, standard-library viewer for the joined-T3 replay report."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def ground_truth_for_report(report, report_path, data_root=None):
    """Materialize verified GT separately, without changing the frozen report."""
    cache = report_path.parent / "ground-truth.json"
    identity = hashlib.sha256(report_path.read_bytes()).hexdigest()
    image_hashes = {row["sample_id"]: row["image_sha256"] for row in report["assembled_inputs"]}
    if data_root is not None:
        from recognition.icdar13_annotations import load_icdar13_recognition_samples
        from diagnostics.joined_t3 import join_crop_transcriptions

        manifest = Path(__file__).resolve().parents[1] / "manifests/icdar13_recognition_test.yaml"
        samples, _ = load_icdar13_recognition_samples(manifest, data_root)
        if {s.sample_id: s.image_sha256 for s in samples} != image_hashes:
            raise ValueError("GT image population does not match the report")
        values = {}
        for sample in samples:
            ordered = join_crop_transcriptions([
                {"candidate_id": str(i), "bbox": list(word.bbox_xyxy), "text": word.transcription}
                for i, word in enumerate(sample.words)
            ])
            values[sample.sample_id] = {
                "image_sha256": sample.image_sha256,
                "annotation_sha256": sample.annotation_sha256,
                "reading_order_text": ordered["text"],
                "words_in_annotation_order": [w.transcription for w in sample.words],
            }
        payload = {"source_report_sha256": identity, "samples": values,
                   "display_order": "geometric line ordering for display only; original word list preserved"}
        if cache.exists():
            if json.loads(cache.read_text(encoding="utf-8")) != payload:
                raise ValueError("existing GT cache mismatch")
        else:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=cache.parent, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            try:
                os.link(temporary, cache)
            finally:
                temporary.unlink()
    if not cache.exists():
        raise FileNotFoundError("GT cache missing: run once with the main Python environment and --data-root ICDAR13_ROOT")
    cached = json.loads(cache.read_text(encoding="utf-8"))
    if (cached["source_report_sha256"] != identity
            or {sid: value["image_sha256"] for sid, value in cached["samples"].items()} != image_hashes):
        raise ValueError("GT cache/report identity mismatch")
    return cached["samples"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=(
        Path(__file__).resolve().parents[1]
        / "artifacts/diagnostics/t3-joined-test-v1/results.json"
    ))
    parser.add_argument("--family", choices=("faithful", "v1_repository", "balanced"), default="balanced")
    parser.add_argument("--caption", choices=("repository", "antirepetition"), default="antirepetition")
    parser.add_argument("--strategy", choices=("selected_candidate_paired", "repository_paired_threshold"), default="selected_candidate_paired")
    parser.add_argument("--sample", help="Show only this sample ID")
    parser.add_argument("--data-root", type=Path, help="Prepare the verified GT display cache (main environment required once)")
    args = parser.parse_args()
    with args.report.open(encoding="utf-8") as stream:
        report = json.load(stream)
    key = "/".join((args.family, args.caption, args.strategy))
    rows = report["per_image"][key]
    if args.sample is not None:
        rows = [row for row in rows if str(row["sample_id"]) == args.sample]
        if not rows:
            parser.error("sample ID not found: " + args.sample)
    gt = ground_truth_for_report(report, args.report, args.data_root)
    print("VARIANTE:", key)
    for row in rows:
        d = row["decision"]
        print("\n" + "=" * 70)
        print("IMMAGINE:", row["sample_id"])
        print("GROUND TRUTH:", gt[str(row["sample_id"])]["reading_order_text"])
        for field in ("T1", "T3", "T2"):
            print(field + (" CAPTION" if field == "T2" else "") + ":", d[field])
        print("S1={:.4f}  S3={:.4f}".format(d["S1"], d["S3"]))
        print("L1={:.4f}  L3={:.4f}".format(d["L1"], d["L3"]))
        print("C={:.4f}  scelto={}  tau={}".format(d["C"], d["semantic_winner"], d["tau"]))
        print("FALLBACK:", d["needs_fallback"])
        print("TESTO FINALE:", row["final_text"])


if __name__ == "__main__":
    main()
