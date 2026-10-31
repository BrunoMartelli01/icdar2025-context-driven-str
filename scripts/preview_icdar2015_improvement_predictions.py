"""Compare saved predictions on validation examples with gains and losses."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from segmentation.data import file_sha256, load_segmentation_manifest
from scripts.train_icdar2015_totaltext_transfer import write_json


def run(args):
    if args.output.exists():raise ValueError("Preserve existing preview")
    reports=[json.loads(path.read_text()) for path in (args.baseline_results,args.candidate_results)]
    rows=[{r["sample_id"]:r for r in report["per_image"]} for report in reports]
    if set(rows[0])!=set(rows[1]):raise ValueError("Paired validation population required")
    delta={key:rows[1][key]["counts"]["alphanumeric_casefold_tp"]-rows[0][key]["counts"]["alphanumeric_casefold_tp"]
        for key in rows[0] if rows[0][key]["ground_truth"]}
    positive=sorted(delta,key=lambda key:(-delta[key],key))[:2]
    negative=next(key for key in sorted(delta,key=lambda key:(delta[key],key)) if key not in positive)
    selected=[*positive,negative]
    samples={s.sample_id:s for s in load_segmentation_manifest(Path(reports[0]["protocol"]["paths"]["manifest"])).samples}
    details=[{r["sample_id"]:r for r in json.loads(Path(report["protocol"]["paths"]["geometry_report"]).read_text())["per_image"]}
        for report in reports]
    canvas=Image.new("RGB",(1260,480*len(selected)+70),"#ededed");draw=ImageDraw.Draw(canvas)
    draw.text((15,12),"Validation prediction inspection / approximate reproduction / cases selected by correct-word TP change",fill="black")
    draw.text((15,32),"Two largest gains and largest remaining loss; this is not an aggregate score or a TEST sample selection.",fill="black")
    audit=[]
    for row,key in enumerate(selected):
        sample=samples[key];path=args.data_root/sample.image_path
        if file_sha256(path)!=sample.image_sha256:raise ValueError("RGB changed")
        with Image.open(path) as im:rgb=im.convert("RGB")
        rgb_draw=ImageDraw.Draw(rgb)
        for word in rows[0][key]["ground_truth"]:rgb_draw.rectangle(word["bbox"],outline="yellow",width=5)
        panels=[rgb];labels=[key+" / care GT in yellow"]
        for index,report in enumerate(reports):
            saved=details[index][key];mask=Path(report["protocol"]["paths"]["mask_dir"])/saved["mask_file"]
            if file_sha256(mask)!=saved["mask_sha256"]:raise ValueError("Prediction mask changed")
            with np.load(mask) as arrays:prediction=arrays["prediction"]
            panels.append(Image.fromarray(prediction.astype(np.uint8)*255).convert("RGB"))
            count=rows[index][key]["counts"]
            labels.append(f"{report['protocol']['branch']}: {count['alphanumeric_casefold_tp']} correct / {count['predictions']} proposals")
        y=70+480*row
        for col,(panel,label) in enumerate(zip(panels,labels)):
            draw.text((col*420+10,y+5),label,fill="black")
            panel=ImageOps.pad(panel,(400,400),method=Image.Resampling.NEAREST,color="black")
            canvas.paste(panel,(col*420+10,y+30))
        draw.text((15,y+445),f"GT words: {len(rows[0][key]['ground_truth'])}; change in correctly localized and recognized words: {delta[key]:+d}",fill="black")
        audit.append({"sample_id":key,"correct_word_tp_change":delta[key]})
    args.output.parent.mkdir(parents=True,exist_ok=True);canvas.save(args.output)
    write_json(args.output.with_suffix(".json"),{"reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "selection_rule":"two largest validation correct-word TP gains and largest loss among other samples with care GT",
        "selected":audit,"baseline_result_sha256":file_sha256(args.baseline_results),
        "candidate_result_sha256":file_sha256(args.candidate_results),"source_sha256":file_sha256(Path(__file__)),
        "image_sha256":file_sha256(args.output)})
    return args.output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-results","candidate-results","data-root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    print(run(parser.parse_args()))


if __name__=="__main__":main()
