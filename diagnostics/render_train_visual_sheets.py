"""Render one self-contained visual sheet per frozen ICDAR13 TRAIN image."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import textwrap
import zipfile

from PIL import Image, ImageDraw, ImageFont, ImageOps

from diagnostics.lavis_captioner_candidates import sha, write_json


EXPECTED_SOURCE_MANIFEST_SHA256 = ""


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def fitted(image: Image.Image, size: tuple[int, int], nearest: bool = False) -> Image.Image:
    result = image.copy().convert("RGB")
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    result.thumbnail(size, method)
    canvas = Image.new("RGB", size, "#eeeeee")
    canvas.paste(result, ((size[0]-result.width)//2, (size[1]-result.height)//2))
    return canvas


def wrap(draw: ImageDraw.ImageDraw, text: str, face, width: int) -> list[str]:
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append(""); continue
        line = words[0]
        for word in words[1:]:
            trial = line + " " + word
            if draw.textbbox((0,0), trial, font=face)[2] <= width:
                line = trial
            else:
                lines.append(line); line = word
        lines.append(line)
    return lines


def text_block(draw, xy, title, text, width, title_face, body_face, color="#111111") -> int:
    x,y = xy
    draw.text((x,y), title, font=title_face, fill="#111111")
    y += title_face.size + 8
    for line in wrap(draw, text, body_face, width):
        draw.text((x,y), line, font=body_face, fill=color)
        y += body_face.size + 6
    return y


def section_height(count: int, columns: int = 4) -> int:
    return 58 + max(1, math.ceil(count / columns)) * 190


def crop_section(canvas, y, title, rows, sample_dir, accent):
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((20,y,canvas.width-20,y+42), fill=accent)
    draw.text((34,y+8), f"{title} ({len(rows)} retained crops)", font=font(22,True), fill="white")
    y += 52
    columns=4; card_w=(canvas.width-50)//columns; card_h=180
    for index,row in enumerate(rows):
        col=index%columns; line=index//columns
        x=20+col*card_w; cy=y+line*card_h
        draw.rectangle((x,cy,x+card_w-10,cy+card_h-10), outline="#999999", width=2, fill="white")
        with Image.open(sample_dir/row["file"]) as opened:
            crop=fitted(opened,(card_w-30,105))
        canvas.paste(crop,(x+10,cy+8))
        transcription=str(row["parseq_text"])
        if len(transcription)>34: transcription=transcription[:31]+"..."
        draw.text((x+10,cy+118),f"#{row['rank']:02d} PARSeq: {transcription}",font=font(16,True),fill="#111111")
        draw.text((x+10,cy+142),f"{row['wordness']} | IoU={float(row['best_gt_iou']):.3f}",font=font(13),fill="#333333")
        draw.text((x+10,cy+160),f"bbox={row['bbox_xyxy_exclusive']}",font=font(11),fill="#555555")
    return y + max(1,math.ceil(len(rows)/columns))*card_h


def render(sample_dir: Path, output: Path) -> dict:
    metadata=json.loads((sample_dir/"metadata.json").read_text())
    gt=metadata["ground_truth_words"];v1=metadata["v1"];faithful=metadata["paper_faithful_algorithm1_approximate"]
    width=1800; top=80; panel_h=430; text_y=top+panel_h+25
    gt_text=" | ".join(f"[{w['index']}] {w['text']}" for w in gt)
    probe=Image.new("RGB",(width,100),"white");pd=ImageDraw.Draw(probe)
    body=font(18); heading=font(22,True)
    gt_lines=wrap(pd,gt_text,body,width-80)
    gt_h=heading.size+15+len(gt_lines)*(body.size+6)+20
    captions=metadata["captions"]
    cap_lines=max(len(wrap(pd,captions["instructblip"],body,810)),len(wrap(pd,captions["florence2-base-ft"],body,810)))
    cap_h=heading.size+15+cap_lines*(body.size+6)+30
    y_crops=text_y+gt_h+cap_h+25
    height=y_crops+section_height(len(v1))+section_height(len(faithful))+35
    canvas=Image.new("RGB",(width,height),"white");draw=ImageDraw.Draw(canvas)
    draw.rectangle((0,0,width,64),fill="#202b3c")
    draw.text((24,14),f"ICDAR13 TRAIN sample {metadata['sample_id']} — frozen visual audit",font=font(30,True),fill="white")
    panels=[
        ("Original RGB", "image.jpg", False),
        ("Ground-truth segmentation mask", "gt_mask_original_binary.png", True),
        ("Predicted mask (epoch-0098, backprojected)", "predicted_mask_original.png", True),
        ("GT vs prediction @128: white TP, green FN, red FP", "gt_vs_prediction_native.png", True),
    ]
    panel_w=(width-50)//4
    for i,(label,name,nearest) in enumerate(panels):
        x=20+i*panel_w
        draw.text((x,74),label,font=font(16,True),fill="#111111")
        with Image.open(sample_dir/name) as opened: shown=fitted(opened,(panel_w-12,375),nearest)
        canvas.paste(shown,(x,102))
    y=text_y
    y=text_block(draw,(30,y),"Ground-truth words and boxes",gt_text,width-60,heading,body)
    y+=15
    left_y=text_block(draw,(30,y),"InstructBLIP caption",captions["instructblip"],830,heading,body)
    right_y=text_block(draw,(930,y),"Florence-2-base-ft caption",captions["florence2-base-ft"],830,heading,body)
    y=max(left_y,right_y)+20
    y=crop_section(canvas,y,"Practical-remap V1 crops + PARSeq",v1,sample_dir,"#087e8b")
    y+=10
    crop_section(canvas,y,"Paper-faithful Algorithm1 approximate crops + PARSeq",faithful,sample_dir,"#8b2f70")
    output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(output,format="JPEG",quality=93,subsampling=0,optimize=True)
    return {"sample_id":metadata["sample_id"],"file":output.name,"sha256":sha(output),"size_bytes":output.stat().st_size,
            "gt_words":len(gt),"v1_crops":len(v1),"paper_faithful_crops":len(faithful),"sheet_shape_hw":[height,width]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-bundle",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--zip-output",type=Path,required=True);a=p.parse_args()
    if a.output_dir.exists() or a.zip_output.exists(): raise FileExistsError("refusing to replace an existing output")
    manifest_path=a.source_bundle/"manifest.json"
    if not manifest_path.exists(): raise FileNotFoundError(manifest_path)
    source=json.loads(manifest_path.read_text())
    if (source["split"],source["images"],source["gt_words"],source["v1_crops"],source["paper_faithful_algorithm1_crops"]) != ("TRAIN",229,849,1203,1761):
        raise RuntimeError("source bundle population mismatch")
    a.output_dir.mkdir(parents=True)
    rows=[]
    for n,sid in enumerate(sorted(x["sample_id"] for x in source["samples"]),1):
        rows.append(render(a.source_bundle/"samples"/sid,a.output_dir/f"train-{sid}.jpg"))
        print(f"[single-sheet bundle] {n}/229 sample={sid}",flush=True)
    if (len(rows),sum(r["gt_words"] for r in rows),sum(r["v1_crops"] for r in rows),sum(r["paper_faithful_crops"] for r in rows)) != (229,849,1203,1761):
        raise RuntimeError("rendered population mismatch")
    readme=("One JPEG per ICDAR13 TRAIN image. Every sheet contains original RGB, GT words/mask, frozen predicted mask, "
            "InstructBLIP and Florence-2-base-ft captions, all V1 crops and all frozen CURRENT_ALG1 approximate crops with PARSeq output. "
            "No new inference was executed. CURRENT_ALG1 is an approximate paper-faithful diagnostic, not strict reproduction.\n")
    (a.output_dir/"README.txt").write_text(readme,encoding="utf-8")
    write_json(a.output_dir/"manifest.json",{"classification":"APPROXIMATE_DIAGNOSTIC","split":"TRAIN","images":229,
        "gt_words":849,"v1_crops":1203,"paper_faithful_algorithm1_crops":1761,"one_sheet_per_image":True,
        "new_inference":False,"source_manifest_sha256":sha(manifest_path),"renderer_sha256":sha(__file__),"sheets":rows})
    a.zip_output.parent.mkdir(parents=True,exist_ok=True);partial=a.zip_output.with_suffix(a.zip_output.suffix+".partial")
    with zipfile.ZipFile(partial,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for path in sorted(a.output_dir.iterdir()): z.write(path,Path(a.output_dir.name)/path.name)
    partial.replace(a.zip_output)
    with zipfile.ZipFile(a.zip_output) as z:
        bad=z.testzip(); entries=len(z.infolist())
    if bad: raise RuntimeError(f"ZIP CRC failure: {bad}")
    print("ZIP",a.zip_output.resolve());print("ZIP_SHA256",sha(a.zip_output));print("ZIP_ENTRIES",entries)


if __name__=="__main__":main()
