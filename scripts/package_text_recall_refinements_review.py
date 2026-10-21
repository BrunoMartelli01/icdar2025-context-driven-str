"""Offline representative review of completed crop refinements; no inference."""

import argparse
from collections import Counter
import csv
import hashlib
import html
from io import BytesIO, StringIO
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image, ImageDraw, ImageFont

from scripts.package_totaltext_highres_residual_pad0_review import _fit, matched_crop_to_word
from scripts.prune_segmentation_checkpoints import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve(); output = args.output.resolve()
    geometry = json.loads((root / "geometry_results.json").read_text())
    recognition = json.loads((root / "recognition_results.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    diagnosis = json.loads((root / "gt_crop_diagnostic.json").read_text())
    if geometry["status"] != recognition["status"] or summary["status"] != "SUCCESS":
        raise ValueError("Complete successful source required")
    if sha256_file(root / "recognition_results.json") != summary["recognition_results_sha256"]:
        raise ValueError("Source result changed")
    candidates = [r for r in summary["rows"] if r["branch"] != "oracle"]
    best = max(candidates, key=lambda r: (r["hmean"], r["correct_parseq_words"], -r["care_proposals"]))
    best_key = best["branch"] + "__selected"
    before = {r["sample_id"]: r for r in recognition["branches"]["reference512__selected"]["baseline"]["per_image"]}
    after = {r["sample_id"]: r for r in recognition["branches"][best_key][best["method"]]["per_image"]}
    images = {r["sample_id"]: r for r in geometry["branches"]["reference512__selected"]["baseline"]["per_image"]}
    def delta(sid):
        a, b = [set(d[sid]["correct_casefold_gt_indices"]) for d in (before, after)]
        return len(b - a), len(a - b)
    ids = sorted(before)
    typical = lambda s: abs(len(after[s]["correct_casefold_gt_indices"]) / max(1, len(after[s]["ground_truth"])) - best["recall"])
    ranks = [sorted(ids, key=lambda s: (-delta(s)[0], delta(s)[1], s)),
             sorted(ids, key=lambda s: (-delta(s)[1], delta(s)[0], s)),
             sorted(ids, key=lambda s: (typical(s), s)),
             sorted(ids, key=lambda s: (-len(before[s]["ground_truth"]), s))]
    categories = ["Parole recuperate", "Parole perse / limiti", "Casi tipici", "Immagini dense"]
    examples = []; used = set()
    for i in range(12):
        category = i % 4; sid = next(s for s in ranks[category] if s not in used); used.add(sid)
        gain, loss = delta(sid)
        examples.append({"sample_id": sid, "category": categories[category], "gained": gain, "lost": loss})
    manifest = {"result_class": "approximate reproduction", "population": "125 reused TRAIN holdout /983 care words",
        "best_observed_arm": best, "selection": "Highest observed Hmean across15 predeclared inference arms; exploratory reused-validation comparison. Oracle excluded.12 diagnostic examples,3/category, not a random sample.",
        "source_summary_sha256": sha256_file(root / "summary.json"), "examples": examples, "files_sha256": {}}
    font = ImageFont.truetype("DejaVuSans.ttf", 17); small = ImageFont.truetype("DejaVuSans.ttf", 15)
    bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    def add(archive, name, data):
        if isinstance(data, str): data = data.encode("utf-8")
        archive.writestr(name, data); manifest["files_sha256"][name] = hashlib.sha256(data).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        table = StringIO(); writer = csv.DictWriter(table, fieldnames=list(summary["rows"][0]))
        writer.writeheader(); writer.writerows(summary["rows"])
        add(archive, "metrics_15_arms_and_separate_oracle.csv", table.getvalue())
        for name in ("summary.json", "REPORT.md", "geometry_protocol.json", "recognition_protocol.json"):
            add(archive, "reports/" + name, (root / name).read_bytes())
        add(archive, "gt_diagnostic_counts.json", json.dumps({"gt_words": diagnosis["gt_words"],
            "gt_box_fixed_pair_correct": diagnosis["gt_box_fixed_pair_correct"], "branches": {n: {m: p["counts"]
            for m, p in methods.items()} for n, methods in diagnosis["branches"].items()}}, indent=2))
        for i, sample in enumerate(examples, 1):
            sid = sample["sample_id"]; row = images[sid]
            if sha256_file(Path(row["image_path"])) != row["image_sha256"]: raise ValueError("Source RGB changed")
            with Image.open(row["image_path"]) as image: rgb = image.convert("RGB")
            predictions = [[p for p in d[sid]["predictions_all"] if p["instance_id"] not in d[sid]["ignored_prediction_ids"]] for d in (before, after)]
            matched = [matched_crop_to_word(p, before[sid]["ground_truth"]) for p in predictions]
            board = Image.new("RGB", (1800, 780 + max(map(len, predictions)) * 25), "white"); draw = ImageDraw.Draw(board)
            draw.text((16, 12), f'{sid} | {sample["category"]} | recuperate {sample["gained"]}, perse {sample["lost"]}', font=bold, fill="black")
            draw.text((16, 46), f'Approximate reproduction | nuovo confronto CPU | {best["branch"]}/{best["method"]}', font=font, fill="#333333")
            draw.text((16, 73), 'Verde = bbox IoU>=0.50 e testo casefold corretto; rosso = errore. GT solo nel pannello blu.', font=font, fill="#555555")
            titles = [f'RGB + GT: {len(before[sid]["ground_truth"])} parole', f'Text+Kernel512: {len(matched[0])} corrette', f'Variante osservata migliore: {len(matched[1])} corrette']
            for column in range(3):
                left = 12 + column * 596; draw.text((left, 115), titles[column], font=bold, fill="black")
                px, py, pw, ph = _fit(board, rgb, (left, 153, left + 580, 716))
                boxes = before[sid]["ground_truth"] if column == 0 else predictions[column - 1]
                for j, p in enumerate(boxes):
                    x1, y1, x2, y2 = p["bbox"]
                    color = "#0060aa" if column == 0 else "#119944" if j in matched[column - 1] else "#ee3333"
                    coords = (px + x1 * pw / rgb.width, py + y1 * ph / rgb.height,
                              px + (x2 - 1) * pw / rgb.width, py + (y2 - 1) * ph / rgb.height)
                    draw.rectangle(coords, outline=color, width=2); draw.text(coords[:2], str(j + 1), font=small, fill=color)
            for column, (preds, matches) in enumerate(zip(predictions, matched)):
                draw.text((16 + column * 896, 745), "Text+Kernel512 / PARSeq" if column == 0 else "Variante / PARSeq", font=font, fill="black")
                for j, p in enumerate(preds):
                    label = f'{j+1:02d} {"OK" if j in matches else "ERRORE"}: {p["text"]!r}'
                    draw.text((16 + column * 896, 778 + j * 25), label[:94], font=small, fill="#117733" if j in matches else "#bb2222")
            stream = BytesIO(); board.save(stream, format="JPEG", quality=91, optimize=True)
            name = f'images/{i:02d}_{sid.replace(":", "__")}.jpg'; add(archive, name, stream.getvalue()); sample["image"] = name
            add(archive, f'samples/{sid.replace(":", "__")}.json', json.dumps({"sample_id": sid, "reference": before[sid], "variant": after[sid],
                "gt_crop_diagnosis": [w for w in diagnosis["branches"][best_key][best["method"]]["per_word"] if w["sample_id"] == sid]}, ensure_ascii=False, indent=2))
            if i == 1: board.save("/tmp/text-recall-refinements-review-preview.jpg")
        cards = "".join(f'<article><h2>{html.escape(s["sample_id"])} — {s["category"]}</h2><a href="{s["image"]}"><img src="{s["image"]}" loading="lazy"></a></article>' for s in examples)
        add(archive, "index.html", '<!doctype html><html lang="it"><meta charset="utf-8"><title>Crop refinements + PARSeq</title><style>body{font:16px system-ui;background:#eee;margin:24px}article{background:white;padding:12px;margin:20px 0}img{width:100%;max-width:1800px}</style><h1>Text+Kernel: nuovi confronti sui crop</h1><p>Approximate reproduction,125 immagini TRAIN holdout /983 parole.15 varianti predefinite; il riferimento e ogni suo crop sono rivalutati su CPU. La variante visualizzata ha il maggiore Hmean osservato su questa validation riutilizzata, senza validazione indipendente. Il risultato GT-box è un diagnostico separato.</p><p><a href="metrics_15_arms_and_separate_oracle.csv">Tutte le metriche</a> | <a href="reports/summary.json">Report JSON</a> | <a href="gt_diagnostic_counts.json">Diagnosi GT crop</a></p>' + cards + '</html>')
        add(archive, "README.txt", "Estrarre e aprire index.html.12 esempi diagnostici:3 per categoria, inclusi guadagni e perdite. Non stimare metriche dai soli esempi. Il CSV include15 bracci predetti e un oracle GT separato. Tutti i risultati sono esplorativi sulla validation TRAIN riutilizzata. Nessun checkpoint incluso.\n")
        add(archive, "package_source.py", Path(__file__).read_bytes())
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with ZipFile(output) as archive:
        if archive.testzip() is not None: raise RuntimeError("ZIP CRC failure")
        for name, digest in manifest["files_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest: raise RuntimeError("ZIP content hash failure")
    output.with_suffix(".zip.sha256").write_text(sha256_file(output) + "  " + output.name + "\n")
    print(json.dumps({"status": "PASS", "zip": str(output), "bytes": output.stat().st_size,
        "sha256": sha256_file(output), "examples": len(examples), "categories": dict(Counter(s["category"] for s in examples))}, indent=2))


if __name__ == "__main__":
    main()
