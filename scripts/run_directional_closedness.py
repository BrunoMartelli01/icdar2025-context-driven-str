"""APPROXIMATE CAL-qualified ownership-closedness readout; reject negative pilots."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from localize.directional_closedness import closedness
from localize.word_multitask import parseq_word_proposals
from recognition.directional_transport_metrics import aggregate, counts
from scripts.evaluate_cached_totaltext_padding import crop_hash
from scripts.run_directional_transport import read, write, setup, model_for, finite_probabilities, OCRCache, memory
from segmentation.data import file_sha256
from segmentation.directional_transport import transform
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.totaltext_word_kernel import expand_word_kernels

REPO=Path(__file__).resolve().parents[1]


def admitted(pool, policy):
    return [p for p in pool if p['confidence']>=policy['confidence'] and
        (p['closedness']['score'] is None or p['closedness']['score']>=policy['closedness'])]


def extract(model,c,source,entries,out,cache_directory,ocr):
    sc=source['resolved_config'];paired=transform(sc,native_size=256,train=False)
    pp=sc['postprocessing'];q=c['fixed_geometry'];records={}
    out.mkdir(parents=True,exist_ok=False)
    model.eval()
    with torch.inference_mode():
        for i,entry in enumerate(entries):
            sid=entry['sample_id'];stem=sid.replace(':','__')
            with Image.open(entry['image_path']) as opened:image=opened.convert('RGB')
            rgb=np.ascontiguousarray(image)
            augmented,mask,_=paired._augment(image,Image.new('L',image.size),None,0)
            tensor,_=paired._finalize(augmented,mask)
            evidence=model.forward_evidence(tensor[None].cuda(),return_edges=True)
            prob=finite_probabilities(evidence['words']);native=prob.shape[-1]
            canvas=np.asarray(letterbox_canvas_mask(image.size,native_size=native),bool)
            labels,_=expand_word_kernels((prob[0]>=q['text_threshold'])&canvas,(prob[1]>=pp['kernel_threshold'])&canvas)
            pool=parseq_word_proposals(labels,image.size,padding=int(q['padding256']*native/256),
                min_area=pp['minimum_area256']*(native/256)**2,max_instances=pp['maximum_instances'])
            for p in pool:p['padded_bbox_original']=p['bbox'];p['bbox']=p['bbox_original']
            # Last transport stage: all configured radii and four directions.
            n=4*len(sc['architecture']['radii'])
            edges=[(z[0,0].sigmoid().cpu().numpy(),valid[0,0].cpu().numpy(),dy,dx)
                   for z,valid,dy,dx in evidence['edges'][-n:]]
            edge_canvas=np.asarray(letterbox_canvas_mask(image.size,native_size=edges[0][0].shape[0]),bool)
            values=closedness(labels,edges,edge_canvas,[p['instance_id'] for p in pool])
            if cache_directory is not None:
                with np.load(cache_directory/(stem+'.npz')) as saved:
                    if not np.array_equal(prob,saved['probabilities']):raise ValueError('Pinned CAL probability replay failed: '+sid)
                cached=read(cache_directory/(stem+'.json'))['pools'][str((q['text_threshold'],q['padding256']))]
                keys=('instance_id','bbox_native','bbox','padded_bbox_original','foreground_area')
                if [[p[k] for k in keys] for p in pool]!=[[p[k] for k in keys] for p in cached]:
                    raise ValueError('CAL crop geometry replay failed: '+sid)
                for p,old in zip(pool,cached):
                    x0,y0,x1,y1=p['padded_bbox_original']
                    if crop_hash(np.ascontiguousarray(rgb[y0:y1,x0:x1]))!=old['crop_array_sha256']:
                        raise ValueError('Cached CAL RGB crop mismatch')
                    p.update(text=old['text'],confidence=old['confidence'],crop_array_sha256=old['crop_array_sha256'])
            else:ocr.image(rgb,pool)
            for p in pool:p['closedness']=values[p['instance_id']]
            records[sid]=pool
            np.savez_compressed(out/(stem+'.npz'),probabilities=prob)
            write(out/(stem+'.json'),dict(sample_id=sid,pool=pool,image_sha256=entry['image_sha256']))
            if (i+1)%25==0:print(out.name,i+1,len(entries),flush=True)
    return records


def scores(records,entries,policies,overlap):
    return [[dict(sample_id=e['sample_id'],**counts(admitted(records[e['sample_id']],p),e['words'],overlap))
             for e in entries] for p in policies]


def best(rows,policies,ids=None):
    metrics=[aggregate([r for r in rr if ids is None or r['sample_id'] in ids]) for rr in rows]
    k=max(range(len(metrics)),key=lambda i:metrics[i]['f1'])
    return k,dict(policy=policies[k],aggregate=metrics[k])


def run(args):
    cp=args.config.resolve()
    if args.mode!='approximate' or REPO/'configs/approximate' not in cp.parents:raise ValueError('Isolated APPROXIMATE config required')
    c=read(cp)
    if c['reproduction_mode']!=args.mode:raise ValueError('Mode mismatch')
    source_root=Path(c['source_run']);source=read(source_root/'protocol.json')
    for key,path in [('source_protocol',source_root/'protocol.json'),('checkpoint',source_root/c['source_arm']/'best.pt')]:
        if file_sha256(path)!=c['input_sha256'][key]:raise ValueError('Source drift: '+key)
    for name,module in list(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if file is None:continue
        path=Path(file).resolve()
        if REPO not in path.parents:continue
        relative=str(path.relative_to(REPO));expected=source['identity']['code_sha256'].get(relative)
        if expected is not None and file_sha256(path)!=expected:raise ValueError('Loaded parent implementation changed: '+relative)
    out=args.output.resolve();out.mkdir(exist_ok=False,parents=True)
    sc=source['resolved_config'];runtime={**sc,'runtime':{**sc['runtime'],'minimum_free_gpu_bytes':c['minimum_free_gpu_bytes']}}
    hardware=setup(runtime)
    protocol=dict(reproduction_mode='approximate',claim_label='approximate reproduction',result_class='CAL-only pilot until qualified',
        resolved_config=c,source_protocol_sha256=c['input_sha256']['source_protocol'],config_sha256=file_sha256(cp),hardware=hardware,
        code_sha256={str(p.relative_to(REPO)):file_sha256(p) for folder in ('models','localize','recognition','segmentation','scripts') for p in (REPO/folder).rglob('*.py')},
        historical_test_exploration=True,selection_uses_test=False,annotations_used_for_prediction=False)
    write(out/'protocol.json',protocol)
    model=model_for(sc,c['source_arm']);payload=torch.load(source_root/c['source_arm']/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(payload['model_state_dict'],strict=True);model.cuda().eval();torch.cuda.reset_peak_memory_stats()
    epoch=payload['epoch'];entries=source['identity']['index']['calibration']
    cache=source_root/c['source_arm']/f'cal-epoch-{epoch:03d}'
    records=extract(model,c,source,entries,out/'calibration',cache,None)
    policies=[dict(confidence=conf,closedness=gate) for gate in c['closedness_thresholds'] for conf in c['confidence_thresholds']]
    rows=scores(records,entries,policies,sc['metric_iou_threshold'])
    baseline_policies=[p for p in policies if p['closedness']==-1.]
    baseline_rows=[rows[i] for i,p in enumerate(policies) if p['closedness']==-1.]
    chosen,selected=best(rows,policies);_,base=best(baseline_rows,baseline_policies)
    ids=sorted([e['sample_id'] for e in entries],key=lambda sid:hashlib.sha256((str(c['seed'])+sid).encode()).hexdigest())
    folds=[set(ids[k::c['folds']]) for k in range(c['folds'])]
    oof,baseline_oof=[],[]
    for held in folds:
        train=set(ids)-held;k,_=best(rows,policies,train);j,_=best(baseline_rows,baseline_policies,train)
        oof.extend(r for r in rows[k] if r['sample_id'] in held)
        baseline_oof.extend(r for r in baseline_rows[j] if r['sample_id'] in held)
    gain=100*(aggregate(oof)['f1']-aggregate(baseline_oof)['f1'])
    gain_fixed=100*(aggregate(oof)['f1']-base['aggregate']['f1'])
    qualified=min(gain,gain_fixed)>=c['minimum_calibration_gain_pp']
    pilot=dict(reproduction_mode='approximate',claim_label='approximate reproduction',result_class='partial-dataset result; CAL-only pilot',
        selected=selected,confidence_only=base,oof=aggregate(oof),oof_confidence_only=aggregate(baseline_oof),
        oof_gain_pp=gain,oof_gain_versus_fixed_cal_baseline_pp=gain_fixed,qualified_for_test=qualified,
        memory=memory(),parameters=sum(p.numel() for p in model.parameters()))
    write(out/'calibration_pilot.json',pilot)
    write(out/'selection_sealed.json',dict(policy=selected['policy'],epoch=epoch,checkpoint_sha256=c['input_sha256']['checkpoint'],qualified=qualified))
    if qualified:
        ocr=OCRCache(sc,out);test=source['identity']['index']['test']
        pred=extract(model,c,source,test,out/'test',None,ocr)
        test_rows=scores(pred,test,[selected['policy']],sc['metric_iou_threshold'])[0]
        write(out/'test/per_sample.json',test_rows)
        write(out/'test/predictions.json',{e['sample_id']:admitted(pred[e['sample_id']],selected['policy']) for e in test})
        write(out/'test/complete.json',dict(reproduction_mode='approximate',claim_label='approximate reproduction',
            result_class='approximate reproduction; exploratory TEST300',aggregate=aggregate(test_rows),policy=selected['policy'],memory=memory()))
        ocr.release()
    write(out/'complete.json',dict(status='SUCCESS',reproduction_mode='approximate',qualified_for_test=qualified,
        result='TEST evaluated after CAL qualification' if qualified else 'Negative CAL pilot; no further TEST evaluation'))
    print(json.dumps(pilot,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['approximate'],required=True)
    args=parser.parse_args()
    try:run(args)
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        write(args.output/'failure.json',dict(reproduction_mode='approximate',error=repr(error),oom=isinstance(error,torch.cuda.OutOfMemoryError)))
        raise
