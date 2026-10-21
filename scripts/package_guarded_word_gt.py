"""Small measured V1/V2/V3 target comparison; never exports RGB OCR crops."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.tighter_word_gt import native_training_owners


def run(directory):
    result=json.loads((directory/'full/results.json').read_text())
    if result['status']!='SUCCESS':raise ValueError('Complete qualification required')
    c=json.loads((directory/'full/protocol.json').read_text())['resolved_config']
    versions={k:c[k] for k in ['reference','compact']}
    samples={k:{s.sample_id:s for s in load_segmentation_manifest(Path(v['manifest']),require_checksums=True).samples}
             for k,v in versions.items()}
    out=directory/'figures';out.mkdir(exist_ok=False)
    font=lambda size:ImageFont.truetype('DejaVuSans.ttf',size)
    canvas=Image.new('RGB',(1280,920),'white');draw=ImageDraw.Draw(canvas)
    draw.text((16,12),'GT V3 | actual TRAIN targets at128 | approximate reproduction',font=font(23),fill='black')
    draw.text((16,46),'Cyan: selected word; grey: other words. No new network training or prediction.',font=font(17),fill='black')
    provenance=[]
    for row,(sid,owner) in enumerate([('total_text:1006',2),('total_text:114',15),('total_text:1009',4)]):
        native={};sources=[]
        for family,v in versions.items():
            s=samples[family][sid];root=Path(v['data_root']);arrays=[]
            for rel,digest,mode in [(s.word_owner_annotation_path,s.word_owner_annotation_sha256,'I'),
                (s.validity_annotation_path,s.validity_annotation_sha256,'L'),(s.separation_annotation_path,s.separation_annotation_sha256,'L')]:
                path=root/rel
                if file_sha256(path)!=digest:raise ValueError('Comparison source changed')
                with Image.open(path) as im:arrays.append(np.asarray(im.convert(mode)).copy())
                sources.append({'path':str(path),'sha256':digest})
            native[family]=native_training_owners(*arrays,(128,128))
        preview=directory/'full/native_preview/owners'/f"{sid.replace(':','__')}.png"
        with Image.open(preview) as im:native['v3']=np.asarray(im.convert('I')).copy()
        s=samples['compact'][sid];root=Path(c['compact']['data_root'])
        annotation_path=(root/s.word_owner_annotation_path).parent.parent/'annotations'/f'{Path(s.word_owner_annotation_path).stem}.json'
        annotation=json.loads(annotation_path.read_text());instance=next(x for x in annotation['instances'] if x['owner_id']==owner)
        xy=np.asarray(instance['polygon_xy']);w,h=annotation['size_wh']
        roi=[max(0,int(np.floor(xy[:,0].min()*128/w))-3),max(0,int(np.floor(xy[:,1].min()*128/h))-3),
             min(128,int(np.ceil(xy[:,0].max()*128/w))+4),min(128,int(np.ceil(xy[:,1].max()*128/h))+4)]
        image_path=root/s.image_path
        if file_sha256(image_path)!=s.image_sha256:raise ValueError('RGB source changed')
        with Image.open(image_path) as im:im=im.convert('RGB')
        im.thumbnail((294,214),Image.Resampling.LANCZOS)
        y=85+row*270;draw.text((16,y),f'{sid}, owner {owner}: {instance["transcription"]}',font=font(19),fill='black')
        left=16+(294-im.width)//2;top=y+42+(214-im.height)//2;canvas.paste(im,(left,top))
        draw.rectangle([left+xy[:,0].min()*im.width/w,top+xy[:,1].min()*im.height/h,
            left+xy[:,0].max()*im.width/w,top+xy[:,1].max()*im.height/h],outline=(240,40,40),width=3)
        counts=[]
        for column,(family,title) in enumerate([('reference','V1'),('compact','V2'),('v3','V3')],1):
            own=native[family];count=int(cv2.connectedComponents((own==owner).astype(np.uint8),connectivity=8)[0]-1);counts.append(count)
            x=16+316*column;draw.text((x,y+27),f'{title}: {count} components',font=font(18),fill='black')
            x0,y0,x1,y1=roi;patch=own[y0:y1,x0:x1];rgb=np.zeros((*patch.shape,3),np.uint8)
            rgb[patch>0]=(75,75,75);rgb[patch==owner]=(0,210,230)
            panel=Image.fromarray(rgb);scale=min(294/panel.width,204/panel.height)
            panel=panel.resize((max(1,round(panel.width*scale)),max(1,round(panel.height*scale))),Image.Resampling.NEAREST)
            canvas.paste(panel,(x+(294-panel.width)//2,y+53+(204-panel.height)//2))
        provenance.append({'sample_id':sid,'owner':owner,'transcription':instance['transcription'],
            'native_components_v1_v2_v3':counts,'source_image_sha256':s.image_sha256,
            'v3_preview_sha256':file_sha256(preview),'source_annotations_sha256':file_sha256(annotation_path),
            'sources':sources})
    canvas.save(out/'v1-v2-v3-connectivity.png')
    (out/'provenance.json').write_text(json.dumps({'claim_label':'approximate reproduction',
        'selection':'same three TRAIN failure examples selected before V3 construction; not a representative sample',
        'examples':provenance,'code_sha256':file_sha256(Path(__file__))},indent=2)+'\n')
    with (directory/'summary.csv').open('w') as f:
        rows=[dict(view=name,**value['stats']) for name,value in result['views'].items()]
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (directory/'unresolved_words.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['view','sample_id','owner','components','pixels','reasons']);writer.writeheader()
        for name in result['views']:
            view=json.loads((directory/'full/views'/f'{name}.json').read_text())
            for image in view['per_image']:
                for word in image['per_word']:
                    if word['unresolved']:
                        writer.writerow({'view':name,'sample_id':image['sample_id'],'owner':word['owner'],
                            'components':word['after_components'],'pixels':word['after_pixels'],'reasons':'|'.join(word['unresolved'])})
    print(json.dumps({'status':'SUCCESS','figure':str(out/'v1-v2-v3-connectivity.png'),'examples':[v['native_components_v1_v2_v3'] for v in provenance]}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    run(p.parse_args().directory)
