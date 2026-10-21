"""Three isolated APPROXIMATE word-evidence tests with persistent phase receipts."""
import argparse
import fcntl
import gc
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader,Subset
from localize.word_multitask import word_instances,parseq_word_proposals
from localize.word_evidence import candidates,associate,adjust_boxes
from models.unet_word_research_v2 import WordResearchV2AGUNet
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _split_indices,_json
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _train_gt,_geometry_match,_correct
from segmentation.data import load_segmentation_manifest,SegmentationDataset
from segmentation.unet_word_research import GlyphWordDataset
from segmentation.totaltext_word_kernel import make_word_kernels
from segmentation.totaltext_image_frequency_augment import remap_letterbox_native_box
from segmentation.word_evidence import supervision,fit_head,predict_head
from segmentation.training import seed_everything,runtime_environment
from segmentation.evaluate import _prepare_cuda_memory_tracking
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE,PARSEQ_CHECKPOINT_SHA256,PARSEQ_RUNTIME_ARCHIVE_SHA256,
    ParseqApproxConfig,ParseqApproxRecognizer)
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash

REPO=Path(__file__).resolve().parents[1]


def read_config(path,mode):
    path=path.resolve()
    if mode!='approximate' or REPO/'configs/approximate' not in path.parents:raise ValueError('Explicit approximate config namespace required')
    c=json.loads(path.read_text())
    if c.get('experiment_kind')!='frozen_word_evidence_three_tests_v1' or c.get('claim_label')!='approximate reproduction' or c.get('reproduction_mode')!=mode:
        raise ValueError('Invalid scientific mode/config')
    source=REPO/c['source_config']
    if sha256_file(source)!=c['source_config_sha256']:raise ValueError('Source config SHA mismatch')
    for k in ('training_count','validation_count','expected_validation_words','export_batch_size','ocr_batch_size','cpu_threads'):
        if type(c[k]) is not int or c[k]<1:raise ValueError('Positive integer '+k+' required')
    t=c['training']; a=c['candidates']
    if t['epochs']<2 or t['hidden']<1 or not 0<t['max_box_delta']<0.5 or t['learning_rate']<=0 or t['weight_decay']<0:
        raise ValueError('Invalid head training config')
    if (not 0<a['glyph_threshold']<1 or not 0<a['kernel_threshold']<1 or not 0<a['association_threshold']<1
            or not .5<=a['owner_majority']<=1 or type(a['nearest_instances']) is not int or a['nearest_instances']<1):raise ValueError('Invalid association configuration')
    return c,json.loads(source.read_text())


def memory(device,stage,c):
    cuda=str(device).startswith('cuda')
    return {'device':str(device),'precision':'float32','model_set':stage,
        'batch_size':c['export_batch_size'] if stage=='frozen segmentation plus isolated glyph head' else c['ocr_batch_size'],
        'input_shape':[c['export_batch_size'],3,512,512] if stage=='frozen segmentation plus isolated glyph head' else [c['ocr_batch_size'],3,32,128],
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if cuda and torch.cuda.is_initialized() else None,
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if cuda and torch.cuda.is_initialized() else None,
        'oom':False}


def gpu_state():
    return subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.strip()


def bind(path,data):
    if path.exists():
        if json.loads(path.read_text())!=data:raise ValueError('Immutable run contract changed: '+str(path))
    else:_json(path,data)


def load_map(path):
    with np.load(path,allow_pickle=False) as saved:return {k:saved[k] for k in saved.files}


def prediction(m,source):return word_instances(m['probabilities'],m['canvas'].astype(bool),source['postprocessing'])[0]


def proposals(labels,m,boxes=None,ignore_labels=None):
    output=parseq_word_proposals(labels,tuple(int(x) for x in m['image_wh']),padding=0,min_area=0,max_instances=None)
    if boxes is not None:
        if len(boxes)!=len(output):raise ValueError('Correction lost proposal identity')
        for p,box in zip(output,boxes):
            p['bbox_native']=list(map(int,box));p['bbox']=remap_letterbox_native_box(p['bbox_native'],tuple(m['image_wh']),native_size=labels.shape[0])
    # Preserve inference even on ignored instances; apply validity only during evaluation.
    support_labels=labels if ignore_labels is None else ignore_labels
    return [{**p,'ignored':not bool(np.any((support_labels==p['instance_id']) & m['validity'].astype(bool)))} for p in output]


def summarize(rows):
    keys=['gt_words','proposals','geometry_correct']
    if all('ocr_correct' in row for row in rows):keys.append('ocr_correct')
    totals={k:sum(r[k] for r in rows) for k in keys}
    kinds=['geometry']+(['ocr'] if 'ocr_correct' in totals else [])
    for kind in kinds:
        tp=totals[kind+'_correct']
        n,g=totals['proposals'],totals['gt_words']
        totals[kind+'_precision']=tp/n if n else 0.
        totals[kind+'_recall']=tp/g if g else 0.
        totals[kind+'_hmean']=2*tp/(n+g) if n+g else 0.
    return totals


def ground_truth(sample,word_root,m):return _train_gt(sample,word_root,tuple(int(v) for v in m['image_wh']))[0]


def evaluate_geometry(records,by_id,word_root,arm):
    rows=[]
    for sid,item in records.items():
        m=item['map'];gt=ground_truth(by_id[sid],word_root,m)
        care=[p for p in item['proposals'] if not p['ignored']]
        matched=_geometry_match(care,gt,.5)
        rows.append({'sample_id':sid,'gt_words':len(gt),'proposals':len(care),'geometry_correct':len(matched),
                     'matched_geometry_gt':list(map(int,matched))})
    return {'arm':arm,'aggregate':summarize(rows),'per_image':rows}


def validation_maps(args,c,source,manifest,val_indices):
    directory=args.output_root/'validation_maps';directory.mkdir(exist_ok=True)
    receipt_path=args.glyph_run/'validation.json'
    if sha256_file(receipt_path)!=c['source_validation_sha256'] or sha256_file(args.control_run/'validation.json')!=c['control_validation_sha256']:raise ValueError('Source validation report changed')
    receipt=json.loads(receipt_path.read_text())
    control=json.loads((args.control_run/'validation.json').read_text())
    originals={r['sample_id']:r for r in receipt['per_image']};reference={r['sample_id']:r for r in control['per_image']}
    expected={manifest.samples[i].sample_id for i in val_indices}
    if set(originals)!=expected or set(reference)!=expected:raise ValueError('Source validation population differs')
    paths={}
    for i in val_indices:
        sample=manifest.samples[i];sid=sample.sample_id;filename=sid.replace(':','__')
        source_path=args.glyph_run/'validation_exports'/filename/'maps.npz'
        if sha256_file(source_path)!=originals[sid]['maps_sha256']:raise ValueError('Source glyph maps changed')
        base_path=args.control_run/'validation_exports'/filename/'maps.npz'
        if sha256_file(base_path)!=reference[sid]['maps_sha256']:raise ValueError('Control map export changed')
        raw=load_map(source_path);base=load_map(base_path)
        if not np.array_equal(raw['probabilities'],base['probabilities']):raise ValueError('Isolated glyph branch changed control word maps')
        labels=prediction(raw,source)
        if not np.array_equal(labels,base['labels']):raise ValueError('Frozen baseline partition replay mismatch')
        image=args.word_root/sample.image_path
        if sha256_file(image)!=sample.image_sha256:raise ValueError('Validation RGB changed')
        with Image.open(image) as opened:wh=opened.size
        data={k:raw[k] for k in ('probabilities','owners','validity','canvas','text_target','glyph_target')}
        data['glyph_probability']=1/(1+np.exp(-np.clip(raw['glyph'][0],-60,60)))
        data['image_wh']=np.asarray(wh,np.int32)
        path=directory/(filename+'.npz')
        if path.exists():
            old=load_map(path)
            if old.keys()!=data.keys() or any(not np.array_equal(old[k],v) for k,v in data.items()):raise ValueError('Changed cached validation evidence')
        else:np.savez_compressed(path,**data)
        paths[sid]=path
    _json(args.output_root/'validation_map_receipt.json',{'status':'PASS','result_class':'approximate reproduction',
        'source_validation_sha256':sha256_file(receipt_path),'control_validation_sha256':sha256_file(args.control_run/'validation.json'),
        'images':len(paths),'maps':{sid:sha256_file(p) for sid,p in paths.items()},'exact_control_partition_replays':len(paths)})
    return paths


def diagnostic(args,c,source,manifest,paths):
    arms={name:{} for name in ('baseline','gt_text_pred_kernel','pred_text_gt_kernel','gt_text_gt_kernel','glyph_gt_gt_kernel')}
    by_id={s.sample_id:s for s in manifest.samples}
    for sid,path in paths.items():
        m=load_map(path);gt_kernel=make_word_kernels(m['owners'],distance_fraction=source['targets']['distance_fraction'])[0]>0
        evidence={'baseline':m['probabilities'],
            'gt_text_pred_kernel':np.stack((m['text_target'],m['probabilities'][1])),
            'pred_text_gt_kernel':np.stack((m['probabilities'][0],gt_kernel)),
            'gt_text_gt_kernel':np.stack((m['text_target'],gt_kernel)),
            'glyph_gt_gt_kernel':np.stack((m['glyph_target'],gt_kernel))}
        for name,p in evidence.items():
            labels=word_instances(p.astype(np.float32),m['canvas'].astype(bool),source['postprocessing'])[0]
            arms[name][sid]={'map':m,'proposals':proposals(labels,m)}
    report={'status':'SUCCESS','reproduction_mode':'approximate','result_class':'approximate reproduction',
        'scope':'TRAIN holdout oracle diagnostics only; GT variants not inference results',
        'arms':{k:evaluate_geometry(v,by_id,args.word_root,k) for k,v in arms.items()}}
    if any(a['aggregate']['gt_words']!=c['expected_validation_words'] for a in report['arms'].values()):raise ValueError('Diagnostic denominator changed')
    _json(args.output_root/'diagnostic_geometry.json',report)
    print('[diagnostic geometry] '+json.dumps({k:v['aggregate']['geometry_correct'] for k,v in report['arms'].items()}),flush=True)
    return arms


def export_training(args,c,source,dataset,indices):
    directory=args.output_root/'training_maps';directory.mkdir(exist_ok=True)
    receipt_path=args.output_root/'training_map_receipt.json'
    if receipt_path.exists():
        r=json.loads(receipt_path.read_text())
        if r['images']!=len(indices) or any(sha256_file(directory/(sid.replace(':','__')+'.npz'))!=h for sid,h in r['maps'].items()):raise ValueError('Training map receipt changed')
        return r
    device=torch.device(args.device)
    resident=gpu_state() if device.type=='cuda' else None
    if device.type=='cuda':
        _prepare_cuda_memory_tracking(device)
        if torch.cuda.mem_get_info(device)[0]<c['minimum_free_gpu_bytes']:raise RuntimeError('Insufficient declared free GPU memory; existing job is not stopped')
    checkpoint=args.glyph_run/'best.pt'
    if sha256_file(checkpoint)!=c['source_checkpoint_sha256']:raise ValueError('Glyph checkpoint changed')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload['epoch']!=c['source_checkpoint_epoch'] or payload['architecture']!='experimental-unet-word-research-v2:glyph_isolated':raise ValueError('Source checkpoint identity changed')
    model=WordResearchV2AGUNet('glyph_isolated',source);model.load_state_dict(payload['model_state_dict'],strict=True);del payload
    control=torch.load(args.control_run/'best.pt',map_location='cpu',weights_only=False)['model_state_dict']
    for key,value in model.state_dict().items():
        if not key.startswith('glyph_private.') and (key not in control or not torch.equal(value,control[key])):raise ValueError('Frozen word tensor mismatch: '+key)
    del control
    model.to(device).eval();hashes={};rows=[]
    try:
        loader=DataLoader(Subset(dataset,indices),batch_size=c['export_batch_size'],shuffle=False,num_workers=0)
        with torch.inference_mode():
            for batch in loader:
                ev=model.forward_evidence(batch['image'].to(device))
                p=ev['words'].sigmoid().cpu().numpy();g=ev['glyph'].sigmoid().cpu().numpy()
                for j,sid in enumerate(batch['sample_id']):
                    with Image.open(batch['image_path'][j]) as opened:wh=opened.size
                    data={'probabilities':p[j],'glyph_probability':g[j,0],'owners':batch['owners_native'][j].numpy(),
                        'validity':batch['validity'][j,0].numpy().astype(bool),'canvas':batch['canvas'][j].numpy().astype(bool),
                        'text_target':batch['mask'][j,0].numpy().astype(bool),'glyph_target':batch['glyph_mask'][j,0].numpy(),
                        'image_wh':np.asarray(wh,np.int32)}
                    data['owners']=data['owners'].copy();data['owners'][~data['validity'] | ~data['text_target']]=0
                    path=directory/(sid.replace(':','__')+'.npz')
                    if path.exists():
                        old=load_map(path)
                        if old.keys()!=data.keys() or any(not np.array_equal(old[k],v) for k,v in data.items()):raise ValueError('Interrupted export replay differs: '+sid)
                    else:np.savez_compressed(path,**data)
                    hashes[sid]=sha256_file(path)
                    rows.append({'sample_id':sid,'surviving_words':int(len(np.unique(data['owners'][data['owners']>0])))})
                if len(hashes)%40==0 or len(hashes)==len(indices):
                    print('[frozen export] '+str(len(hashes))+'/'+str(len(indices)),flush=True)
                    _json(args.output_root/'progress.json',{'stage':'frozen_training_export','completed':len(hashes),'total':len(indices),'status':'RUNNING','result_class':'approximate reproduction'})
        receipt={'status':'PASS','result_class':'approximate reproduction','images':len(hashes),'maps':hashes,
                 'memory':memory(device,'frozen segmentation plus isolated glyph head',c),
                 'external_gpu_processes_before':resident,'external_gpu_processes_after':gpu_state() if device.type=='cuda' else None,'per_image':rows}
        _json(receipt_path,receipt);return receipt
    finally:
        del model;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()


def train_modules(args,c,source,manifest,train_indices):
    dest=args.output_root/'trained_heads.json'
    seal=args.output_root/'head_completion.json'
    if dest.exists():
        expected=json.loads(seal.read_text())
        if sha256_file(dest)!=expected['heads_sha256'] or sha256_file(args.output_root/'training_map_receipt.json')!=expected['training_maps_sha256']:raise ValueError('Trained head provenance changed')
        return json.loads(dest.read_text())
    pair_x=[];pair_y=[];box_x=[];box_y=[];boxes=[];truth=[];neighbors=[];rows=[]
    for i in train_indices:
        sid=manifest.samples[i].sample_id;m=load_map(args.output_root/'training_maps'/(sid.replace(':','__')+'.npz'))
        labels=prediction(m,source);can=candidates(m['probabilities'],m['glyph_probability'],labels,m['canvas'],c['candidates'])
        sup=supervision(can,labels,m['owners'],m['validity'].astype(bool),c['candidates'])
        pk=sup['pair_keep'];bk=sup['box_keep']
        pair_x.append(can['pair_features'][pk]);pair_y.append(sup['pair_target'][pk]);box_x.append(can['box_features'][bk]);box_y.append(sup['box_target'][bk])
        boxes.append(can['boxes'][bk]);truth.append(sup['reference_boxes'][bk]);neighbors.extend([n for keep,n in zip(bk,sup['neighbor_boxes']) if keep])
        rows.append({'sample_id':sid,'association_pairs':int(pk.sum()),'ambiguous_pairs':int((~pk).sum()),'crop_targets':int(bk.sum()),'unsupervised_crop_proposals':int((~bk).sum())})
        if len(rows)%100==0:print('[supervision] '+str(len(rows))+'/'+str(len(train_indices)),flush=True)
    if len(rows)!=c['training_count']:raise ValueError('Training population incomplete')
    _json(args.output_root/'head_training_population.json',{'status':'PASS','result_class':'approximate reproduction','per_image':rows,'training_images':len(rows),'augmentation':'none: frozen canonical predictions, declared calibration experiment'})
    cfg=c['training'];result={}
    result['affinity']=fit_head(np.concatenate(pair_x),np.concatenate(pair_y),cfg,kind='affinity')
    _json(args.output_root/'affinity_head.json',result['affinity']);print('[affinity head] trained',flush=True)
    result['crop']=fit_head(np.concatenate(box_x),np.concatenate(box_y),cfg,kind='crop',boxes=np.concatenate(boxes),truth=np.concatenate(truth),neighbors=neighbors)
    _json(args.output_root/'crop_head.json',result['crop']);_json(dest,result)
    _json(seal,{'status':'SUCCESS','result_class':'approximate reproduction','heads_sha256':sha256_file(dest),'training_maps_sha256':sha256_file(args.output_root/'training_map_receipt.json')})
    print('[crop head] trained',flush=True)
    return result


def inference_arms(args,c,source,paths,heads):
    arms={'fragment_association':{},'crop_completeness':{}}
    for sid,path in paths.items():
        m=load_map(path);labels=prediction(m,source)
        can=candidates(m['probabilities'],m['glyph_probability'],labels,m['canvas'],c['candidates'])
        scores=predict_head(heads['affinity'],can['pair_features'])
        extended,audit=associate(labels,can['fragments'],can['pair_ids'],scores,c['candidates']['association_threshold'])
        arms['fragment_association'][sid]={'map':m,'proposals':proposals(extended,m,ignore_labels=labels),'audit':audit}
        corrections=predict_head(heads['crop'],can['box_features'])
        boxes=adjust_boxes(can['boxes'],corrections,labels.shape)
        arms['crop_completeness'][sid]={'map':m,'proposals':proposals(labels,m,boxes=boxes),'audit':{'corrected_boxes':int(np.any(boxes!=can['boxes'],axis=1).sum())}}
    return arms


def score_ocr(args,c,manifest,arms):
    cache=args.output_root/'ocr';cache.mkdir(exist_ok=True)
    by_id={s.sample_id:s for s in manifest.samples};jobs={};frozen={}
    # Freeze proposals/crops before requesting annotations.
    for arm,records in arms.items():
        frozen[arm]=[]
        for sid,item in records.items():
            sample=by_id[sid];image=args.word_root/sample.image_path
            if sha256_file(image)!=sample.image_sha256:raise ValueError('Changed evaluation RGB')
            with Image.open(image) as opened:rgb=np.asarray(opened.convert('RGB'))
            predictions=[]
            for p in item['proposals']:
                box=list(map(int,p['bbox']));x1,y1,x2,y2=box
                if not 0<=x1<x2<=rgb.shape[1] or not 0<=y1<y2<=rgb.shape[0]:raise ValueError('Invalid original RGB crop')
                h=crop_array_hash(np.ascontiguousarray(rgb[y1:y2,x1:x2]));record={**p,'bbox':box,'crop_rgb_sha256':h}
                predictions.append(record)
                if not (cache/(h+'.json')).exists():jobs[h]=(image,box)
            frozen[arm].append({'sample_id':sid,'image_sha256':sample.image_sha256,'proposals':predictions,'audit':item.get('audit',{})})
    frozen_path=args.output_root/'frozen_evaluation_proposals.json';bind(frozen_path,{'result_class':'approximate reproduction','arms':frozen})
    if sha256_file(args.parseq_checkpoint)!=PARSEQ_CHECKPOINT_SHA256 or sha256_file(args.parseq_archive)!=PARSEQ_RUNTIME_ARCHIVE_SHA256:raise ValueError('Qualified PARSeq artifacts changed')
    recognizer=None
    try:
        if jobs:
            recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,checkpoint=args.parseq_checkpoint,
                device='cpu',runtime_source_type=PARSEQ_ARCHIVE_SOURCE,runtime_source_archive=args.parseq_archive))
            pending=sorted(jobs.items())
            for start in range(0,len(pending),c['ocr_batch_size']):
                part=pending[start:start+c['ocr_batch_size']];crops=[]
                for h,(path,box) in part:
                    with Image.open(path) as opened:rgb=np.asarray(opened.convert('RGB'))
                    x1,y1,x2,y2=box;crop=np.ascontiguousarray(rgb[y1:y2,x1:x2])
                    if crop_array_hash(crop)!=h:raise ValueError('OCR crop identity changed')
                    crops.append(crop)
                outputs=recognizer.recognize_candidate_batch(crops)
                if len(outputs)!=len(part):raise RuntimeError('OCR dropped a sample')
                for (h,_),prediction in zip(part,outputs):_json(cache/(h+'.json'),{'crop_rgb_sha256':h,'prediction':prediction.to_dict(),
                    'checkpoint_sha256':PARSEQ_CHECKPOINT_SHA256,'archive_sha256':PARSEQ_RUNTIME_ARCHIVE_SHA256,'device':'cpu','batch_size':c['ocr_batch_size'],'dtype':'float32','decode_ar':True,'refine_iters':1})
                if start%(c['ocr_batch_size']*10)==0:print('[PARSeq CPU] '+str(min(start+len(part),len(pending)))+'/'+str(len(pending)),flush=True)
    finally:
        if recognizer is not None:recognizer.release()
    results={}
    for arm,samples in frozen.items():
        rows=[]
        for row in samples:
            sid=row['sample_id'];predictions=[]
            for p in row['proposals']:
                cached=json.loads((cache/(p['crop_rgb_sha256']+'.json')).read_text())
                if (cached['checkpoint_sha256']!=PARSEQ_CHECKPOINT_SHA256 or cached['archive_sha256']!=PARSEQ_RUNTIME_ARCHIVE_SHA256
                    or cached['crop_rgb_sha256']!=p['crop_rgb_sha256'] or cached['device']!='cpu' or cached['batch_size']!=c['ocr_batch_size'] or not cached['decode_ar'] or cached['refine_iters']!=1):raise ValueError('OCR cache provenance changed')
                predictions.append({**p,'text':cached['prediction']['raw_text']})
            m=arms[arm][sid]['map'];gt=ground_truth(by_id[sid],args.word_root,m);care=[p for p in predictions if not p['ignored']]
            geom=_geometry_match(care,gt,.5);exact=_correct(care,gt,normalize=False)
            rows.append({**row,'ground_truth':gt,'predictions':predictions,'gt_words':len(gt),'proposals':len(care),
                         'geometry_correct':len(geom),'ocr_correct':len(exact),'matched_geometry_gt':list(map(int,geom)),'matched_ocr_gt':list(map(int,exact))})
        if len(rows)!=c['validation_count'] or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Incomplete evaluation population')
        aggregate=summarize(rows)
        if aggregate['gt_words']!=c['expected_validation_words']:raise ValueError('Care word denominator changed')
        results[arm]={'aggregate':aggregate,'per_image':rows,'deployment_eligible':arm in ('baseline','fragment_association','crop_completeness')}
    baseline={r['sample_id']:r for r in results['baseline']['per_image']}
    for arm,part in results.items():
        gained=[];lost=[]
        for row in part['per_image']:
            old=set(baseline[row['sample_id']]['matched_ocr_gt']);new=set(row['matched_ocr_gt'])
            gained.extend([row['sample_id'],int(i)] for i in sorted(new-old));lost.extend([row['sample_id'],int(i)] for i in sorted(old-new))
        part['paired_vs_baseline']={'gained':len(gained),'lost':len(lost),'gained_words':gained,'lost_words':lost}
    return {'status':'SUCCESS','reproduction_mode':'approximate','result_class':'approximate reproduction','metric_protocol':c['metric_protocol'],
            'arms':results,'ocr_memory':memory('cpu','PARSeq only',c),'oracle_warning':'GT-cross arms are diagnostics, not inference methods or paper results'}


def run(args):
    args.output_root.mkdir(parents=True,exist_ok=True)
    c,source=read_config(args.config,args.mode)
    for root in (args.word_root,args.glyph_root,args.glyph_run,args.control_run,args.parseq_source):
        if args.output_root==root.resolve() or root.resolve() in args.output_root.parents:raise ValueError('Output must be isolated from scientific inputs')
    torch.set_num_threads(c['cpu_threads']);seed_everything(c['seed'],deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    _json(args.output_root/'progress.json',{'stage':'preflight','status':'RUNNING','result_class':'approximate reproduction'})
    current_stage='preflight'
    try:
        if sha256_file(args.glyph_run/'run_manifest.json')!=c['source_receipt_sha256'] or sha256_file(args.control_run/'best.pt')!=c['control_checkpoint_sha256']:raise ValueError('Frozen source receipt/checkpoint changed')
        manifests=[root/source['train_manifest'] for root in (args.word_root,args.glyph_root)]
        for path,key in zip(manifests,('word_manifest_sha256','glyph_manifest_sha256')):
            if sha256_file(path)!=source[key]:raise ValueError('Dataset manifest changed')
        word,glyph=[load_segmentation_manifest(p,require_checksums=True) for p in manifests]
        train_ids,val_ids,split_sha=_split_indices(word,c['seed'],c['validation_count'])
        if split_sha!=source['validation_ids_sha256'] or len(train_ids)!=c['training_count']:raise ValueError('Training/validation split changed')
        contract={'result_class':'approximate reproduction','reproduction_mode':'approximate','resolved_config':c,
            'inputs':{k:str(getattr(args,k).resolve()) for k in ('word_root','glyph_root','glyph_run','control_run','parseq_source','parseq_checkpoint','parseq_archive')},
            'source_sha256':{str(p.relative_to(REPO)):sha256_file(p) for folder in ('models','localize','segmentation','recognition','scripts') for p in (REPO/folder).glob('*.py')},
            'split_sha256':split_sha,'feature_export_device':args.device,'environment':runtime_environment()}
        bind(args.output_root/'run_contract.json',contract)
        paths=validation_maps(args,c,source,word,val_ids)
        current_stage='diagnostic'
        _json(args.output_root/'progress.json',{'stage':current_stage,'status':'RUNNING','result_class':'approximate reproduction'})
        arms=diagnostic(args,c,source,word,paths)
        if args.action=='diagnostic':return
        current_stage='training_map_export'
        if sha256_file(REPO/source['source_config'])!=source['source_config_sha256']:raise ValueError('Frozen preprocessing configuration changed')
        dataset=GlyphWordDataset(SegmentationDataset(word,args.word_root,transform=_transform(json.loads((REPO/source['source_config']).read_text()),'letterbox',train=False)),glyph,args.glyph_root,**source['targets'])
        export_training(args,c,source,dataset,train_ids)
        current_stage='head_training'
        _json(args.output_root/'progress.json',{'stage':current_stage,'status':'RUNNING','result_class':'approximate reproduction'})
        heads=train_modules(args,c,source,word,train_ids)
        current_stage='head_inference';arms.update(inference_arms(args,c,source,paths,heads))
        current_stage='PARSeq'
        _json(args.output_root/'progress.json',{'stage':current_stage,'status':'RUNNING','result_class':'approximate reproduction'})
        report=score_ocr(args,c,word,arms)
        report['run_contract_sha256']=sha256_file(args.output_root/'run_contract.json')
        _json(args.output_root/'results.json',report)
        lines=['# Three word-evidence tests — approximate reproduction','','TRAIN125/983 care words, reused holdout. GT-cross rows are diagnostic only.','',
            '| Arm | Localized /983 | Correct OCR /983 | Care proposals | OCR P | OCR Hmean | Gained/lost |','| --- | ---: | ---: | ---: | ---: | ---: | --- |']
        for name,part in report['arms'].items():
            a=part['aggregate'];p=part['paired_vs_baseline']
            lines.append(f"| {name} | {a['geometry_correct']} | {a['ocr_correct']} | {a['proposals']} | {a['ocr_precision']:.2%} | {a['ocr_hmean']:.2%} | +{p['gained']}/-{p['lost']} |")
        (args.output_root/'comparison.md').write_text('\n'.join(lines)+'\n')
        _json(args.output_root/'progress.json',{'stage':'complete','status':'SUCCESS','result_class':'approximate reproduction'})
        print('\n'.join(lines),flush=True)
    except Exception as exc:
        mem=memory(args.device,'frozen segmentation plus isolated glyph head',c);mem['oom']=isinstance(exc,torch.cuda.OutOfMemoryError)
        _json(args.output_root/'progress.json',{'stage':current_stage,'status':'FAILED','result_class':'approximate reproduction'})
        _json(args.output_root/'failure.json',{'status':'FAILED','result_class':'approximate reproduction','stage':current_stage,'error_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'memory':mem})
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['approximate'],required=True);p.add_argument('--config',type=Path,required=True)
    for name in ('word-root','glyph-root','glyph-run','control-run','parseq-source','parseq-checkpoint','parseq-archive','output-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--action',choices=['all','diagnostic'],default='all')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    args.output_root=args.output_root.resolve()
    args.output_root.mkdir(parents=True,exist_ok=True)
    with (args.output_root/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)
