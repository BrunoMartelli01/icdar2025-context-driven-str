"""Render measured target maps with nearest interpolation, without model output."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from segmentation.icdar2015_improvements import native_target_audit_maps


def run(args):
    if args.output.exists():raise ValueError("Preserve existing preview")
    config=json.loads(args.config.read_text())
    selection=[("icdar2015",args.icdar_root,key) for key in args.icdar_ids]
    selection += [("total_text",args.totaltext_root,key) for key in args.totaltext_ids]
    canvas=Image.new("RGB",(1600,430*len(selection)+50),"#eeeeee")
    draw=ImageDraw.Draw(canvas)
    draw.text((15,15),"TRAIN word targets / approximate reproduction / colours = word owners / black = background",fill="black")
    for row,(dataset,root,key) in enumerate(selection):
        base=root/"datasets"/dataset/"train"
        with Image.open(base/"images"/(key+".jpg")) as im:rgb=im.convert("RGB")
        maps=[]
        for folder in ("masks","validity","separators","owners"):
            with Image.open(base/folder/(key+".png")) as im:maps.append(im.convert("I" if folder=="owners" else "L"))
        panels=[rgb];titles=[dataset+":"+key+" RGB"]
        arrays=[np.asarray(maps[-1]).copy()]
        for name in ("square512","square768"):
            _,mask,owners,_=native_target_audit_maps(*maps,config["resolutions"][name],threshold=config["probability_threshold"])
            owners[~mask]=0;arrays.append(owners)
        for owners in arrays:
            colours=np.zeros((*owners.shape,3),np.uint8)
            for identity in np.unique(owners):
                if identity:colours[owners==identity]=[(int(identity)*67+71)%200+55,
                    (int(identity)*113+19)%200+55,(int(identity)*47+137)%200+55]
            panels.append(Image.fromarray(colours))
        titles += ["Source word targets","RGB512 / native256","RGB768 / native384"]
        for col,(panel,title) in enumerate(zip(panels,titles)):
            y=50+row*430
            draw.text((col*400+10,y+5),title,fill="black")
            panel=ImageOps.contain(panel,(380,380),method=Image.Resampling.NEAREST)
            canvas.paste(panel,(col*400+10+(380-panel.width)//2,y+30+(380-panel.height)//2))
    args.output.parent.mkdir(parents=True,exist_ok=True);canvas.save(args.output)
    return args.output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","icdar-root","totaltext-root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--icdar-ids",nargs="+",required=True)
    parser.add_argument("--totaltext-ids",nargs="+",required=True)
    print(run(parser.parse_args()))


if __name__=="__main__":main()
