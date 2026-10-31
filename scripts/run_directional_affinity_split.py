"""APPROXIMATE learned-affinity instance decoder; CAL cross-validation gates TEST."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch

from localize.directional_affinity_split import affinity_split
from localize.word_multitask import parseq_word_proposals
from recognition.directional_transport_metrics import aggregate, counts
from scripts.run_directional_transport import read, write, setup, model_for, finite_probabilities, OCRCache, memory
from segmentation.data import file_sha256
from segmentation.directional_transport import transform
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.totaltext_word_kernel import expand_word_kernels

REPO=Path(__file__).resolve().parents[1]


def extract(model,c,parent,entries,out,ocr,variants,cache=None):
    sc=parent['resolved_config'];q=c['fixed_geometry'];pp=sc['postprocessing']
    paired=transform(sc,native_size=256,train=False)
    output={str(t):{} for t in variants};out.mkdir(parents=True,exist_ok=False)
    audits=[]
    with torch.inference_mode():
        for index,entry in enumerate(entries):
            sid=entry['sample_id'];stem=sid.replace(':','__')
            if file_sha256(Path(entry['image_path']))!=entry['image_sha256']:raise ValueError('Image bytes changed')
            with Image.open(entry['image_path']) as opened:image=opened.convert('RGB')
            rgb=np.ascontiguousarray(image)
            aug,mask,_=paired._augment(image,Image.new('L',image.size),None,0)
            tensor,_=paired._finalize(aug,mask)
            evidence=model.forward_evidence(tensor[None].cuda(),return_edges=True)
            probability=finite_probabilities(evidence['words']);native=probability.shape[-1]
            if cache is not None:
                with np.load(cache/(stem+'.npz')) as saved:
                    if not np.array_equal(probability,saved['probabilities']):raise ValueError('Parent probability replay failed')
            canvas=np.asarray(letterbox_canvas_mask(image.size,native_size=native),bool)
            labels,_=expand_word_kernels((probability[0]>=q['text_threshold'])&canvas,(probability[1]>=pp['kernel_threshold'])&canvas)
            n=4*len(sc['architecture']['radii'])
            edges=[(z[0,0].sigmoid().cpu().numpy(),v[0,0].cpu().numpy(),dy,dx) for z,v,dy,dx in evidence['edges'][-n:]]
            edge_canvas=np.asarray(letterbox_canvas_mask(image.size,native_size=edges[0][0].shape[0]),bool)
            pools={};maps={'probabilities':probability}
            for threshold in variants:
                partition,audit=affinity_split(labels,edges,edge_canvas,threshold=threshold,**c['split_guards'])
                proposals=parseq_word_proposals(partition,image.size,padding=int(q['padding256']*native/256),
                    min_area=pp['minimum_area256']*(native/256)**2,max_instances=pp['maximum_instances'])
                for p in proposals:p['padded_bbox_original']=p['bbox'];p['bbox']=p['bbox_original']
                ocr.image(rgb,proposals)
                output[str(threshold)][sid]=proposals;pools[str(threshold)]=proposals
                maps['labels_'+str(threshold)]=partition
                audits.append(dict(sample_id=sid,threshold=threshold,**audit))
            write(out/(stem+'.json'),dict(sample_id=sid,image_sha256=entry['image_sha256'],pools=pools))
            np.savez_compressed(out/(stem+'.npz'),**maps)
            if (index+1)%25==0:print(out.name,index+1,len(entries),flush=True)
    write(out/'split_audit.json',audits)
    return output


def score(records,entries,policy,overlap):
    return [dict(sample_id=e['sample_id'],**counts([p for p in records[str(policy['threshold'])][e['sample_id']]
        if p['confidence']>=policy['confidence']],e['words'],overlap)) for e in entries]


def choose(rows,policies,ids=None):
    results=[aggregate([r for r in rr if ids is None or r['sample_id'] in ids]) for rr in rows]
    k=max(range(len(results)),key=lambda j:results[j]['f1'])
    return k,dict(policy=policies[k],aggregate=results[k])


def run(args):
    path=args.config.resolve()
    if args.mode!='approximate' or REPO/'configs/approximate' not in path.parents:raise ValueError('Isolated APPROXIMATE mode/config required')
    c=read(path);source=Path(c['source_run']);parent=read(source/'protocol.json')
    if c['reproduction_mode']!=args.mode:raise ValueError('Mode mismatch')
    for name,p in [('source_protocol',source/'protocol.json'),('checkpoint',source/c['source_arm']/'best.pt'),('ocr_cache',source/'ocr.jsonl')]:
        if file_sha256(p)!=c['input_sha256'][name]:raise ValueError('Source hash drift: '+name)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sc=parent['resolved_config'];runtime={**sc,'runtime':{**sc['runtime'],'minimum_free_gpu_bytes':c['minimum_free_gpu_bytes']}}
    hardware=setup(runtime)
    write(out/'protocol.json',dict(reproduction_mode='approximate',claim_label='approximate reproduction',
        result_class='CAL-only pilot until qualified',resolved_config=c,hardware=hardware,config_sha256=file_sha256(path),
        source_sha256=c['input_sha256'],code_sha256={str(p.relative_to(REPO)):file_sha256(p) for folder in ('models','segmentation','recognition','localize','scripts') for p in (REPO/folder).rglob('*.py')},
        historical_test_exploration=True,selection_uses_test=False,annotations_used_for_prediction=False))
    shutil.copyfile(source/'ocr.jsonl',out/'ocr.jsonl');ocr=OCRCache(sc,out);inherited=len(ocr.rows)
    model=model_for(sc,c['source_arm']);payload=torch.load(source/c['source_arm']/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(payload['model_state_dict'],strict=True);model.cuda().eval();torch.cuda.reset_peak_memory_stats()
    entries=parent['identity']['index']['calibration'];cache=source/c['source_arm']/f'cal-epoch-{payload["epoch"]:03d}'
    variants=[None]+c['affinity_thresholds']
    records=extract(model,c,parent,entries,out/'calibration',ocr,variants,cache)
    policies=[dict(threshold=t,confidence=confidence) for t in variants for confidence in c['confidence_thresholds']]
    rows=[score(records,entries,p,sc['metric_iou_threshold']) for p in policies]
    _,best=choose(rows,policies)
    base_p=[p for p in policies if p['threshold'] is None];base_r=rows[:len(base_p)]
    _,baseline=choose(base_r,base_p)
    expected=read(source/c['source_arm']/'selection.json')['aggregate']
    if baseline['aggregate']!=expected:raise ValueError('No-split CAL baseline does not reproduce source')
    ids=sorted([e['sample_id'] for e in entries],key=lambda s:hashlib.sha256((str(c['seed'])+s).encode()).hexdigest())
    predicted=[];base_predicted=[]
    for fold in range(c['folds']):
        held=set(ids[fold::c['folds']]);train=set(ids)-held
        i,_=choose(rows,policies,train);j,_=choose(base_r,base_p,train)
        predicted.extend(r for r in rows[i] if r['sample_id'] in held)
        base_predicted.extend(r for r in base_r[j] if r['sample_id'] in held)
    value=aggregate(predicted);base_value=aggregate(base_predicted)
    gain=100*(value['f1']-base_value['f1']);gain_fixed=100*(value['f1']-baseline['aggregate']['f1'])
    qualified=best['policy']['threshold'] is not None and min(gain,gain_fixed)>=c['minimum_calibration_gain_pp']
    report=dict(reproduction_mode='approximate',claim_label='approximate reproduction',result_class='partial-dataset result; CAL-only pilot',
        selected=best,baseline=baseline,oof=value,oof_baseline=base_value,oof_gain_pp=gain,
        oof_gain_against_fixed_baseline_pp=gain_fixed,qualified_for_test=qualified,memory=memory(),
        parameters=sum(p.numel() for p in model.parameters()),inherited_ocr_entries=inherited,native_ocr_new_calls=ocr.new_calls)
    write(out/'calibration_pilot.json',report)
    write(out/'selection_sealed.json',dict(policy=best['policy'],qualified=qualified,checkpoint_sha256=c['input_sha256']['checkpoint']))
    if qualified:
        entries=parent['identity']['index']['test']
        records=extract(model,c,parent,entries,out/'test',ocr,[None,best['policy']['threshold']])
        tests={}
        for name,policy in [('graph',best['policy']),('no_graph_same_confidence',dict(threshold=None,confidence=best['policy']['confidence']))]:
            rr=score(records,entries,policy,sc['metric_iou_threshold']);tests[name]=dict(policy=policy,aggregate=aggregate(rr))
            write(out/'test'/(name+'_per_sample.json'),rr)
        write(out/'test/report.json',dict(reproduction_mode='approximate',claim_label='approximate reproduction',
            result_class='approximate reproduction; exploratory TEST300',results=tests,memory=memory()))
    ocr.release()
    write(out/'complete.json',dict(status='SUCCESS',reproduction_mode='approximate',qualified_for_test=qualified,
        result='TEST evaluated' if qualified else 'Negative CAL pilot; TEST not evaluated'))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['approximate'],required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:run(args)
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        write(args.output/'failure.json',dict(reproduction_mode='approximate',error=repr(error),oom=isinstance(error,torch.cuda.OutOfMemoryError)))
        raise
