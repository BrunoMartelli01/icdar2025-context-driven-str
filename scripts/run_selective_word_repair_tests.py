"""Four isolated APPROXIMATE selective word-repair tests on qualified frozen inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import random
import subprocess
import time
import traceback

import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
import torch

from recognition.selective_word_repair import (BoundaryHead, annotation_text_label,
    append_compatible, assess, boundary_layout, boundary_pairs, compact, conditional_features,
    conflicts, exact_matching, local_variants, recovery_label, render_boundaries,
    routed_candidates, triage)
from recognition.text_reconstruction import box_iou
from localize.occlusion_four_tests import features, ocr_features
from localize.teacher_student_tests import CandidateScorer, compatible_selection
from scripts.run_teacher_student_tests import baseline_predictions
from scripts.run_text_reconstruction_tests import sha, digest, read, write, annotations

REPO = Path(__file__).resolve().parents[1]
PHASES = ('fit', 'calibration', 'evaluation', 'test')


def log(root, stage, **values):
    record = dict(stage=stage, pid=os.getpid(), **values)
    write(root/'status.json', record)
    print(stage, json.dumps(values), flush=True)


def checked_config(path, mode):
    path = Path(path).resolve()
    if mode != 'approximate' or REPO/'configs/approximate' not in path.parents:
        raise ValueError('Explicit APPROXIMATE namespace required')
    c = read(path)
    if c['reproduction_mode'] != mode or c['experiment_kind'] != 'selective_word_repair_tests_v1':
        raise ValueError('Unsupported experiment contract')
    for name, expected in c['input_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Scientific input changed: '+name)
    if c['budget'] != 20 or c['matching_iou'] != .5:
        raise ValueError('This paired qualification requires source budget20 / IoU0.5')
    return c


def setup(c):
    torch.set_num_threads(c['runtime']['cpu_threads'])
    torch.set_num_interop_threads(1)
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    return dict(python=platform.python_version(), executable=os.sys.executable,
        packages={k:importlib.metadata.version(k) for k in ('torch','transformers','numpy','scipy','scikit-learn','Pillow')},
        deterministic_algorithms=True, tf32=False, seed=c['seed'])


def gpu_start(c, label):
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != c['runtime']['cublas_workspace_config']:
        raise RuntimeError('Set the configured CUBLAS_WORKSPACE_CONFIG')
    if c['runtime']['device'] != 'cuda:0' or c['runtime']['precision'] != 'float32' or not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA FP32 device unavailable')
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    if free < c['runtime']['minimum_free_bytes']:
        raise RuntimeError('Insufficient free GPU memory; other jobs are preserved')
    torch.cuda.reset_peak_memory_stats()
    return dict(label=label, gpu=torch.cuda.get_device_name(0), total_gpu_bytes=total,
        free_before_bytes=free, other_processes=subprocess.check_output(
            ['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True))


def gpu_end(record, label, batch, shape, started):
    torch.cuda.synchronize()
    return dict(**record, model_set=label, precision='float32', device='cuda:0', batch_size=batch,
        input_shape=shape, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), seconds=time.monotonic()-started, oom=False)


def aggregate(rows, metric):
    result = {k:sum(r[metric][k] for r in rows) for k in ('correct','predictions','gt_words')}
    t,n,g = (result[k] for k in ('correct','predictions','gt_words'))
    result.update(fp=n-t, fn=g-t, precision=t/n if n else 0., recall=t/g if g else 0.,
                  f1=2*t/(n+g) if n+g else 0.)
    return result


def report(samples, predictions, c, reference=None):
    rows = []
    for s in samples:
        sid = s['sample_id']; words = annotations(s)
        value = dict(sample_id=sid, **assess(predictions[sid], words, c['matching_iou']))
        if reference is not None:
            left = assess(reference[sid], words, c['matching_iou'])
            for metric in ('spatial','lexical'):
                key = 'matched_gt_indices' if metric == 'spatial' else 'matched'
                if metric == 'spatial':
                    a,b = set(left[metric][key]),set(value[metric][key])
                else:
                    a,b = Counter(left[metric][key]),Counter(value[metric][key])
                value[metric]['recovered'] = len(b-a) if metric == 'spatial' else sum((b-a).values())
                value[metric]['damaged'] = len(a-b) if metric == 'spatial' else sum((a-b).values())
        rows.append(value)
    result = dict(images=len(rows), per_image=rows,
                  aggregate={m:aggregate(rows,m) for m in ('spatial','lexical')})
    for metric in ('spatial','lexical'):
        if reference is not None:
            for key in ('recovered','damaged'):
                result['aggregate'][metric][key] = sum(r[metric][key] for r in rows)
    return result


def save_arm(root, phase, name, samples, predictions, c, reference):
    write(root/f'{phase}_{name}_predictions.json', predictions)
    value = report(samples,predictions,c,reference)
    write(root/f'{phase}_{name}_results.json',value)
    return value


def prepare(root, c, runtime, smoke):
    root.mkdir(parents=True, exist_ok=False)
    source = Path(c['paths']['source_run'])
    protocol, manifest = read(source/'protocol.json'), read(source/'final_artifact_manifest.json')
    if protocol['source_code_sha256']['recognition/parseq.py'] != sha(REPO/'recognition/parseq.py'):
        raise ValueError('The current PARSeq adapter differs from the frozen cache')
    cache = {}
    for line in (source/'ocr_0.jsonl').open():
        item = json.loads(line); key = item['crop_array_sha256']
        if key in cache and cache[key] != item:
            raise ValueError('Conflicting source recognition records')
        cache[key] = item
    text_source = read(Path(c['paths']['text_run'])/'observations.json')
    text_source = {phase:{s['sample_id']:s for s in rows} for phase,rows in text_source.items()}
    restoration = {x['sample_id']:x for x in read(c['paths']['restoration_manifest'])['restored']}
    records, used_hashes = {}, {}
    for phase in PHASES:
        records[phase] = {}
        entries = protocol['index'][phase]
        if smoke:
            entries = entries[:c['smoke_images'][phase]]
        for entry in entries:
            sid = entry['sample_id']
            relative = 'exports/'+phase+'/'+sid.replace(':','__')+'.json'
            path = source/relative
            expected = manifest[relative]
            if isinstance(expected,dict): expected=expected['sha256']
            if sha(path) != expected:
                raise ValueError('Source export changed: '+relative)
            records[phase][sid] = read(path); used_hashes[str(path)] = expected
    # Reproduce source budgeted selection, never fit a replacement baseline.
    raw, available = {}, {}
    for phase in PHASES:
        raw[phase],available[phase] = baseline_predictions(
            dict(source_run=str(source)),records[phase],cache,c['budget'])
    frozen = torch.load(Path(c['paths']['filter_run'])/'selector/pointwise.pt',map_location='cpu',weights_only=True)
    model = CandidateScorer(frozen['mean'].numel(),32)
    model.load_state_dict(frozen['state_dict'],strict=True); model.eval()
    old_threshold = read(Path(c['paths']['filter_run'])/'selector/selection_frozen.json')['pointwise']
    data, rgb_roles = {}, {}
    for phase in PHASES:
        data[phase] = []
        for sid,r in records[phase].items():
            original = text_source[phase][sid]
            if original['image_sha256'] != r['image_sha256']:
                raise ValueError('Cross-source RGB identity differs')
            image = Path(r['image_path'])
            if sha(image) != r['image_sha256']:
                if sid not in restoration:
                    raise ValueError('Changed RGB without qualified restoration: '+sid)
                image = Path(restoration[sid]['isolated_copy'])
                if sha(image) != r['image_sha256']:
                    raise ValueError('Restored RGB no longer matches')
            pool, vectors = [], {}
            for p in r['arms']['mk']['pool']:
                if p['id'] not in available[phase][sid]: continue
                o = cache[p['crop_array_sha256']]
                pool.append(dict(**p,text=o['raw_text']))
                vectors[p['id']] = r['features']['mk'][p['id']]+ocr_features(o)
            if len({p['crop_array_sha256'] for p in pool}) > c['budget']:
                raise ValueError('Pre-OCR budget exceeded')
            with torch.inference_mode():
                scores = model((torch.tensor([vectors[p['id']] for p in pool])-frozen['mean'])/frozen['scale']).sigmoid().tolist() if pool else []
            baseline = compatible_selection(pool,dict(zip([p['id'] for p in pool],scores)),old_threshold,c['compatibility_iou'])
            raw_rows = [dict(**p,text=cache[p['crop_array_sha256']]['raw_text']) for p in raw[phase][sid]]
            sample = dict(sample_id=sid,image_path=str(image),image_hw=r['image_hw'],image_sha256=r['image_sha256'],
                annotation_path=r['annotation_path'],annotation_sha256=original['annotation_sha256'],
                maps_path=r['maps_path'],maps_sha256=r['maps_sha256'],pool=pool,vectors=vectors,
                baseline_raw=raw_rows,baseline_filter=baseline,text_blocks=original['blocks'])
            if sha(sample['annotation_path']) != sample['annotation_sha256']:
                raise ValueError('Changed annotation')
            data[phase].append(sample)
            rgb_roles.setdefault(r['image_sha256'],[]).append((phase,sid))
    crossings = [v for v in rgb_roles.values() if len({p for p,s in v})>1]
    if any(any(p in ('calibration','evaluation') for p,s in v) for v in crossings):
        raise ValueError('Identical RGB crosses development roles')
    if not smoke:
        known = sorted(s for v in crossings for p,s in v if p=='test')
        if known != sorted(c['known_duplicate_test_ids']): raise ValueError('Unknown RGB duplicates')
    write(root/'samples.json',data)
    with (root/'source_ocr.pkl').open('wb') as stream: pickle.dump(cache,stream)
    critical = ['recognition/selective_word_repair.py','scripts/run_selective_word_repair_tests.py',
        'recognition/text_reconstruction.py','recognition/parseq.py','localize/occlusion_four_tests.py',
        'localize/teacher_student_tests.py','scripts/run_occlusion_four_tests.py',
        'scripts/run_teacher_student_tests.py','scripts/run_text_reconstruction_tests.py',
        'scripts/evaluate_totaltext_v1_findcontours_parseq.py','scripts/run_mask_crop_budget.py']
    plan = dict(config=c,config_sha256=digest(c),runtime=runtime,smoke=smoke,
        source_code_sha256={k:sha(REPO/k) for k in critical},input_exports_sha256=used_hashes,
        samples_sha256=sha(root/'samples.json'),source_ocr_sha256=sha(root/'source_ocr.pkl'),
        populations={p:len(rows) for p,rows in data.items()},rgb_crossings=crossings,
        claim_label='approximate reproduction',independent_validation=False)
    write(root/'protocol.json',plan);write(root/'resolved_configuration.json',c)
    for phase in ('calibration',):
        reference={s['sample_id']:s['baseline_raw'] for s in data[phase]}
        measured=report(data[phase],reference,c)
        old=read(Path(c['paths']['filter_run'])/'selector'/(phase+'_results.json'))
        for kind in ('raw','filter'):
            predictions={s['sample_id']:s['baseline_'+kind] for s in data[phase]}
            current=save_arm(root,phase,'baseline_'+kind,data[phase],predictions,c,reference)
            expected=old['baseline_budget20' if kind=='raw' else 'pointwise']['aggregate']
            if not smoke and any(current['aggregate']['spatial'][k]!=expected[v] for k,v in
                                 [('correct','correct'),('predictions','proposals'),('gt_words','gt_words')]):
                raise ValueError('Source baseline exact replay failed: '+phase+'/'+kind)
    log(root,'PREPARED',populations=plan['populations'])
    return data


def load_prepared(root,c,smoke):
    p=read(root/'protocol.json')
    if p['config_sha256']!=digest(c) or p['smoke']!=smoke:raise ValueError('Changed run configuration/scope')
    for k,v in p['source_code_sha256'].items():
        if sha(REPO/k)!=v:raise ValueError('Changed critical source: '+k)
    if sha(root/'samples.json')!=p['samples_sha256']:raise ValueError('Prepared inputs changed')
    if sha(root/'source_ocr.pkl')!=p['source_ocr_sha256']:raise ValueError('Prepared OCR changed')
    return read(root/'samples.json')


def fit_binary(x,y,c,label):
    x=np.asarray(x,dtype=np.float32);y=np.asarray(y,dtype=np.int64)
    if not len(y):raise ValueError('No training examples: '+label)
    model=HistGradientBoostingClassifier(random_state=c['seed'],**c['classifier']).fit(x,y)
    return model,dict(examples=len(y),positives=int(y.sum()),features=x.shape[1],classes=model.classes_.tolist(),
                      status='TRAINED' if len(set(y))==2 else 'SINGLE_CLASS_EMPIRICAL_CONSTANT')


def probability(model,x):
    x=np.asarray(x,dtype=np.float32)
    if not len(x):return []
    if 1 not in model.classes_:return [0.]*len(x)
    return model.predict_proba(x)[:,list(model.classes_).index(1)].tolist()


def selectors_stage(root,c,data):
    directory=root/'selectors';directory.mkdir(exist_ok=False)
    fit_words={s['sample_id']:annotations(s) for s in data['fit']}
    xs,yt,yc,xa,ya,folds=[],[],[],[],[],[]
    per_image={}
    for i,s in enumerate(data['fit']):
        sid=s['sample_id'];words=fit_words[sid];base=s['baseline_filter']
        start=len(xs)
        for p in s['pool']:
            x=conditional_features(p,base,s['vectors'][p['id']]);xs.append(x);folds.append(i%c['triage']['oof_folds'])
            yt.append(annotation_text_label(p,words,c['triage']['annotation_fraction']))
            yc.append(int(exact_matching([p],words,c['matching_iou'])['correct']>0))
            if p['id'] not in {q['id'] for q in base} and not any(conflicts(p,q,c['compatibility_iou']) for q in base):
                xa.append(x);ya.append(recovery_label(base,base+[p],words,c['matching_iou'])['beneficial'])
        per_image[sid]=(start,len(xs))
    text,mt=fit_binary(xs,yt,c,'text-presence proxy')
    correct,mc=fit_binary(xs,yc,c,'exact-read correctness')
    addition,ma=fit_binary(xa,ya,c,'conditional addition utility')
    models=dict(text=text,correct=correct,addition=addition)
    with (directory/'models.pkl').open('wb') as stream:pickle.dump(models,stream)
    write(directory/'training.json',dict(text=mt,correct=mc,addition=ma,
        label_note='Care-annotation overlap and exact-word matching are separate labels; no F1-only cleanup positive for addition.'))
    oof=np.zeros((len(xs),2));folds=np.asarray(folds);xarray=np.asarray(xs,dtype=np.float32)
    for fold in range(c['triage']['oof_folds']):
        training=folds!=fold;held=~training
        if not held.any():continue
        for col,target in enumerate((yt,yc)):
            m,meta=fit_binary(xarray[training],np.asarray(target)[training],c,'OOF triage')
            oof[held,col]=probability(m,xarray[held])
    scores={}
    for phase in PHASES:
        scores[phase]={}
        for s in data[phase]:
            vectors=[conditional_features(p,s['baseline_filter'],s['vectors'][p['id']]) for p in s['pool']]
            t,q=(oof[slice(*per_image[s['sample_id']])].T.tolist() if phase=='fit'
                 else (probability(text,vectors),probability(correct,vectors)))
            a=probability(addition,vectors)
            scores[phase][s['sample_id']]={p['id']:[t[i],q[i],a[i]] for i,p in enumerate(s['pool'])}
    write(directory/'scores.json',scores)
    cal=data['calibration'];reference={s['sample_id']:s['baseline_filter'] for s in cal}
    trials=[]
    for threshold in c['addition_thresholds']:
        predictions={s['sample_id']:append_compatible(s['baseline_filter'],s['pool'],
            {k:v[2] for k,v in scores['calibration'][s['sample_id']].items()},threshold,c['compatibility_iou']) for s in cal}
        value=report(cal,predictions,c,reference)
        trials.append(dict(threshold=threshold,aggregate=value['aggregate']))
    add_best=max(trials,key=lambda r:(r['aggregate']['spatial']['f1'],-r['aggregate']['spatial']['predictions']))
    raw={s['sample_id']:s['baseline_raw'] for s in cal};triage_trials=[]
    for tt in c['triage']['text_thresholds']:
        for ct in c['triage']['correct_thresholds']:
            policy=dict(text_threshold=tt,correct_threshold=ct)
            predictions={s['sample_id']:[p for p in s['baseline_raw'] if triage(p,*scores['calibration'][s['sample_id']][p['id']][:2],policy)!='reject'] for s in cal}
            value=report(cal,predictions,c,raw)
            triage_trials.append(dict(policy=policy,aggregate=value['aggregate']))
    valid=[r for r in triage_trials if r['aggregate']['spatial']['damaged']<=c['triage']['damage_limit']]
    if not valid:raise RuntimeError('Identity triage missing')
    triage_best=max(valid,key=lambda r:(r['aggregate']['spatial']['f1'],r['aggregate']['spatial']['correct']))
    selection=dict(addition_threshold=add_best['threshold'],triage=triage_best['policy'],uses='calibration only')
    write(directory/'calibration.json',dict(additions=trials,triage=triage_trials))
    write(directory/'selection_frozen.json',selection)
    for phase in PHASES:
        pp={};routes={}
        for s in data[phase]:
            sid=s['sample_id'];local=scores[phase][sid]
            pp[sid]=[p for p in s['baseline_raw'] if triage(p,*local[p['id']][:2],selection['triage'])!='reject']
            routes[sid]=[p['id'] for p in routed_candidates(pp[sid],{k:v[:2] for k,v in local.items()},
                            c['triage']['route_policy'],c['triage']['maximum_reread_parents'])]
        write(directory/(phase+'_triage_predictions.json'),pp)
        write(directory/(phase+'_reread_routes.json'),routes)
        if phase=='fit':continue
        ref={s['sample_id']:s['baseline_raw'] for s in data[phase]}
        save_arm(directory,phase,'triage_defer',data[phase],pp,c,ref)
        additions={s['sample_id']:append_compatible(s['baseline_filter'],s['pool'],
            {k:v[2] for k,v in scores[phase][s['sample_id']].items()},selection['addition_threshold'],c['compatibility_iou']) for s in data[phase]}
        ref={s['sample_id']:s['baseline_filter'] for s in data[phase]}
        save_arm(directory,phase,'additions',data[phase],additions,c,ref)
        # A matched single-correctness control distinguishes deferred rejection.
        hard={s['sample_id']:[p for p in s['baseline_raw'] if scores[phase][s['sample_id']][p['id']][1]>=selection['triage']['correct_threshold']] for s in data[phase]}
        save_arm(directory,phase,'triage_hard',data[phase],hard,c,{s['sample_id']:s['baseline_raw'] for s in data[phase]})
    write(directory/'complete.json',dict(status='SUCCESS',selection=selection))
    log(root,'SELECTORS_COMPLETE',training=dict(text=mt,correct=mc,addition=ma),selection=selection)


def text_report(samples, predictions, c, reference):
    from recognition.text_reconstruction import lexical_metrics, lexical_changes
    rows=[]
    for s in samples:
        sid=s['sample_id'];words=annotations(s)
        value=lexical_metrics([p['text'] for p in predictions[sid]],words)
        value.update(lexical_changes([p['text'] for p in reference[sid]],
                                    [p['text'] for p in predictions[sid]],words))
        value.update(fp=value['predictions']-value['correct'],fn=value['gt_words']-value['correct'])
        rows.append(dict(sample_id=sid,lexical=value))
    result=dict(images=len(rows),per_image=rows,aggregate=dict(lexical=aggregate(rows,'lexical')))
    for key in ('recovered','damaged'):
        result['aggregate']['lexical'][key]=sum(r['lexical'][key] for r in rows)
    return result


def boundaries_stage(root,c,data,smoke):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    directory=root/'boundaries';directory.mkdir(exist_ok=False)
    fit_words={s['sample_id']:annotations(s) for s in data['fit']}
    pairs=boundary_pairs(data['fit'],fit_words,c['boundaries'],random.Random(c['seed']))
    if smoke:pairs=pairs[:c['smoke_pairs']]
    if not pairs:raise ValueError('No clean boundary supervision')
    write(directory/'training_pairs.json',pairs)
    # Contradictory target segmentations are counted, not silently resolved.
    targets={}
    for p in pairs:targets.setdefault(p['source'],set()).add(p['target'])
    conflict_count=sum(len(v)>1 for v in targets.values())
    memory_record=gpu_start(c,'frozen ByT5 encoder');started=time.monotonic();encoder_shapes=set()
    tokenizer=AutoTokenizer.from_pretrained(c['paths']['byt5_snapshot'],local_files_only=True)
    model,loading=AutoModelForSeq2SeqLM.from_pretrained(c['paths']['byt5_snapshot'],local_files_only=True,
        torch_dtype=torch.float32,output_loading_info=True)
    if any(loading[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('Unexpected ByT5 pretrained loading state: '+str(loading))
    encoder=model.get_encoder();del model;gc.collect()
    encoder.eval().cuda()
    for p in encoder.parameters():p.requires_grad_(False)
    encoder_digest_before=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in encoder.parameters())).hexdigest()
    dimensions=encoder.config.d_model+3

    def encode(texts):
        output=[]
        batch=c['runtime']['encoder_batch_size']
        for start in range(0,len(texts),batch):
            group=texts[start:start+batch];layouts=[boundary_layout(t) for t in group]
            inputs=tokenizer([v[0] for v in layouts],padding=True,return_tensors='pt')
            encoder_shapes.add(tuple(inputs['input_ids'].shape))
            with torch.inference_mode():
                hidden=encoder(**{k:v.cuda() for k,v in inputs.items()}).last_hidden_state.cpu().numpy()
            for i,(chars,old_gaps,indices) in enumerate(layouts):
                if not indices:
                    output.append(np.empty((0,dimensions),np.float32));continue
                raw=chars.encode('utf-8')
                if tokenizer(chars)['input_ids'][:-1]!=[v+3 for v in raw]:
                    raise ValueError('Byte/token offset contract changed')
                context=hidden[i,indices]
                extra=np.asarray([[old_gaps[j],(j+1)/len(chars),len(chars)/40] for j in range(len(indices))],np.float32)
                output.append(np.concatenate([context,extra],axis=1))
        return output

    train_features=encode([p['source'] for p in pairs])
    offsets=np.cumsum([0]+[len(x) for x in train_features])
    x=np.concatenate(train_features);y=np.asarray([v for p in pairs for v in p['labels']],np.float32)
    if len(x)!=len(y):raise RuntimeError('Gap supervision is not aligned')
    mean=x.mean(0);scale=x.std(0).clip(c['boundaries']['minimum_feature_scale'])
    np.save(directory/'fit_features.npy',x);np.save(directory/'fit_labels.npy',y)
    write(directory/'fit_feature_offsets.json',offsets.tolist())
    phase_features={}
    for phase in ('calibration','evaluation','test'):
        items=[(s['sample_id'],b) for s in data[phase] for b in s['text_blocks']]
        eligible=[(sid,b) for sid,b in items if len(compact(b['text']).encode('utf-8'))<=c['boundaries']['max_bytes']]
        encoded=encode([b['text'] for sid,b in eligible])
        phase_features[phase]={sid:{} for sid,b in items}
        for (sid,b),value in zip(eligible,encoded):phase_features[phase][sid][b['id']]=value
    encoder_digest_after=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in encoder.parameters())).hexdigest()
    if encoder_digest_before!=encoder_digest_after:raise RuntimeError('Frozen encoder changed')
    memory_encoder=gpu_end(memory_record,'Frozen pinned pretrained ByT5 encoder only; decoder released',
        c['runtime']['encoder_batch_size'],sorted(map(list,encoder_shapes)),started)
    del encoder;gc.collect();torch.cuda.empty_cache()
    head_record=gpu_start(c,'boundary head training');started=time.monotonic()
    torch.manual_seed(c['seed']);head=BoundaryHead(dimensions,c['boundaries']['hidden']).cuda()
    optimizer=torch.optim.AdamW(head.parameters(),lr=c['boundaries']['learning_rate'],weight_decay=c['boundaries']['weight_decay'])
    tx=torch.from_numpy((x-mean)/scale).cuda();ty=torch.from_numpy(y).cuda()
    rng=torch.Generator().manual_seed(c['seed']);history=[]
    epochs=1 if smoke else c['boundaries']['epochs']
    for epoch in range(1,epochs+1):
        losses=[];head.train();order=torch.randperm(len(y),generator=rng)
        for start in range(0,len(y),c['boundaries']['batch_size']):
            ids=order[start:start+c['boundaries']['batch_size']].cuda()
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(head(tx[ids]),ty[ids],
                pos_weight=torch.tensor(c['boundaries']['positive_loss_weight'],device='cuda:0'))
            if not loss.isfinite():raise FloatingPointError('Nonfinite boundary loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),c['boundaries']['gradient_clip'],error_if_nonfinite=True);optimizer.step()
            losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),updates=len(losses)))
        log(root,'BOUNDARY_TRAIN',**history[-1])
    head.eval();scores={}
    with torch.inference_mode():
        for phase,items in phase_features.items():
            scores[phase]={}
            for sid,blocks in items.items():
                scores[phase][sid]={}
                for identity,z in blocks.items():
                    values=head(torch.from_numpy((z-mean)/scale).cuda()).sigmoid().cpu().tolist() if len(z) else []
                    scores[phase][sid][identity]=values
    write(directory/'scores.json',scores);write(directory/'history.json',history)
    torch.save(dict(state_dict=head.cpu().state_dict(),mean=mean,scale=scale,
        dimensions=dimensions,hidden=c['boundaries']['hidden'],encoder_digest=encoder_digest_before),directory/'head.pt')
    memory_head=gpu_end(head_record,'Trained boundary MLP + AdamW; frozen encoder already unloaded',
        c['boundaries']['batch_size'],[len(y),dimensions],started)
    write(directory/'memory.json',[memory_encoder,memory_head])
    write(directory/'training.json',dict(pairs=len(pairs),kinds=dict(Counter(p['kind'] for p in pairs)),
        gap_positions=len(y),positive_gaps=int(y.sum()),source_target_conflicts=conflict_count,
        encoder_loading=loading,encoder_digest_before=encoder_digest_before,encoder_digest_after=encoder_digest_after,
        trainable_parameters=sum(p.numel() for p in head.parameters())))
    del head,optimizer,tx,ty,x;gc.collect();torch.cuda.empty_cache()

    def predict(phase,threshold):
        result={}
        for s in data[phase]:
            rows=[]
            for b in s['text_blocks']:
                pp=scores[phase][s['sample_id']].get(b['id'])
                text=(b['text'] if threshold>1 or pp is None else render_boundaries(
                    b['text'],pp,threshold,c['boundaries']['maximum_output_words']))
                rows.append(dict(id=b['id'],bbox=b['bbox'],text=text,source_text=b['text']))
            result[s['sample_id']]=rows
        return result

    baseline={phase:{s['sample_id']:[dict(id=b['id'],bbox=b['bbox'],text=b['text']) for b in s['text_blocks']]
                     for s in data[phase]} for phase in ('calibration','evaluation','test')}
    trials=[]
    for threshold in c['boundaries']['thresholds']:
        values=text_report(data['calibration'],predict('calibration',threshold),c,baseline['calibration'])
        trials.append(dict(threshold=threshold,aggregate=values['aggregate']))
    valid=[r for r in trials if r['aggregate']['lexical']['damaged']<=c['boundaries']['damage_limit']]
    selection=max(valid,key=lambda r:(r['aggregate']['lexical']['correct'],r['aggregate']['lexical']['f1'],r['threshold']))
    write(directory/'calibration.json',trials);write(directory/'selection_frozen.json',selection)
    old_cal=read(Path(c['paths']['text_run'])/'denoiser_calibration.json')
    for phase in ('calibration','evaluation','test'):
        source_old=old_cal if phase=='calibration' else {r['sample_id']:r['arms'] for r in read(Path(c['paths']['text_run'])/(phase+'_per_image.json'))}
        old={};boundary_only={}
        for s in data[phase]:
            sid=s['sample_id'];groups=source_old[sid]['denoiser_preserve']
            if isinstance(groups,dict):groups=groups['groups']
            old[sid]=[dict(id=g['parents'][0],bbox=g['bbox'],text=' '.join(g['texts'])) for g in groups]
            original={b['id']:b for b in s['text_blocks']}
            boundary_only[sid]=[dict(p,text=p['text'] if compact(p['text']).casefold()==compact(original[p['id']]['text']).casefold()
                                    else original[p['id']]['text']) for p in old[sid]]
        arms=dict(baseline=baseline[phase],byt5_previous=old,byt5_previous_boundaries_only=boundary_only,
                  gap_head_raw=predict(phase,c['boundaries']['raw_threshold']),gap_head_selected=predict(phase,selection['threshold']))
        for name,pred in arms.items():
            write(directory/f'{phase}_{name}_predictions.json',pred)
            write(directory/f'{phase}_{name}_results.json',text_report(data[phase],pred,c,baseline[phase]))
    write(directory/'complete.json',dict(status='SUCCESS',selection=selection))
    log(root,'BOUNDARIES_COMPLETE',selection=selection)


def reread_stage(root,c,data):
    from recognition.parseq import ParseqApproxConfig, ParseqApproxRecognizer, PARSEQ_ARCHIVE_SOURCE
    from scripts.evaluate_cached_totaltext_padding import crop_hash
    directory=root/'reread';directory.mkdir(exist_ok=False)
    with (root/'source_ocr.pkl').open('rb') as stream:cache=pickle.load(stream)
    predictions={phase:read(root/'selectors'/(phase+'_triage_predictions.json')) for phase in PHASES}
    routes={phase:read(root/'selectors'/(phase+'_reread_routes.json')) for phase in PHASES}
    routing_scores=read(root/'selectors/scores.json')
    record=gpu_start(c,'selective local PARSeq rereading');started=time.monotonic()
    p=c['paths']
    recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(p['parseq_source']),
        checkpoint=Path(p['parseq_checkpoint']),device=c['runtime']['device'],runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
        runtime_source_archive=Path(p['parseq_archive'])))
    cache_path=directory/'new_ocr.jsonl';cache_path.touch(exist_ok=False)
    model_input_hw=list(recognizer.model.hparams.img_size)
    data_variants={};costs={};new_inferred=0;cache_hits=0;max_batch=0;input_shapes=set()
    for phase in PHASES:
        data_variants[phase]={};costs[phase]=[]
        for number,s in enumerate(data[phase],1):
            sid=s['sample_id'];base=predictions[phase][sid];by={p['id']:p for p in base}
            if sha(s['image_path'])!=s['image_sha256'] or sha(s['maps_path'])!=s['maps_sha256']:
                raise ValueError('Changed rereading RGB/probability evidence')
            with Image.open(s['image_path']) as im:rgb=np.ascontiguousarray(im.convert('RGB'))
            with np.load(s['maps_path']) as z:prob=z['probabilities'][0]
            variants=[];pending={};requested=set()
            for identity in routes[phase][sid]:
                parent=by[identity]
                proposals=local_variants(parent,s['pool'],set(by),s['image_hw'],prob,c['reread'])
                for j,variant in enumerate(proposals):
                    children=[]
                    for k,box in enumerate(variant['boxes']):
                        crop=np.ascontiguousarray(rgb[box[1]:box[3],box[0]:box[2]])
                        key=crop_hash(crop);requested.add(key);input_shapes.add(crop.shape)
                        child=dict(id=f"reread:{identity}:{j}:{k}",bbox=box,padded_bbox_original=box,
                            parents=[identity],method='reread',crop_array_sha256=key)
                        children.append(child)
                        if key not in cache:pending.setdefault(key,crop)
                        else:cache_hits+=1
                    others=[q for q in base if q['id']!=identity]
                    if any(conflicts(child,q,c['compatibility_iou']) for child in children for q in others):
                        continue
                    variants.append(dict(id=f"variant:{identity}:{j}",parent=identity,kind=variant['kind'],children=children))
            keys=list(pending)
            for start in range(0,len(keys),c['runtime']['ocr_batch_size']):
                batch=keys[start:start+c['runtime']['ocr_batch_size']];max_batch=max(max_batch,len(batch))
                outputs=recognizer.recognize_candidate_batch([pending[k] for k in batch])
                if len(outputs)!=len(batch):raise RuntimeError('PARSeq result cardinality mismatch')
                with cache_path.open('a') as stream:
                    for key,result in zip(batch,outputs):
                        value=dict(**result.to_dict(),crop_array_sha256=key,source='new qualified PARSeq inference')
                        cache[key]=value;stream.write(json.dumps(value,ensure_ascii=False)+'\n');new_inferred+=1
            for variant in variants:
                parent=by[variant['parent']];vectors=[]
                for child in variant['children']:
                    ocr=cache[child['crop_array_sha256']];child['text']=ocr['raw_text']
                    vectors.append(features(child,rgb,prob,text_threshold=c['reread']['feature_text_threshold'])+ocr_features(ocr))
                average=np.mean(vectors,axis=0)
                old=np.asarray(s['vectors'][parent['id']])
                context=conditional_features(parent,base,[])
                variant['features']=[*old.tolist(),*average.tolist(),*(average-old).tolist(),*context,
                    float(variant['kind']=='expand'),float(variant['kind']=='union'),float(variant['kind']=='split'),
                    len(variant['children']),*routing_scores[phase][sid][parent['id']][:2]]
                new_conf=float(np.mean([cache[q['crop_array_sha256']]['mean_token_log_probability']
                    if cache[q['crop_array_sha256']]['mean_token_log_probability'] is not None else -20.
                    for q in variant['children']]))
                old_ocr=cache[parent['crop_array_sha256']]
                variant['confidence_gain']=new_conf-(old_ocr['mean_token_log_probability'] if old_ocr['mean_token_log_probability'] is not None else -20.)
            data_variants[phase][sid]=variants
            initial={p['crop_array_sha256'] for p in s['pool']}
            costs[phase].append(dict(sample_id=sid,admitted_source_unique_crops=len(initial),
                routed_parents=len(routes[phase][sid]),generated_variants=len(variants),
                requested_unique_reread_crops=len(requested),additional_cold_start_crops=len(requested-initial)))
            if number%100==0 or number==len(data[phase]):
                log(root,'REREAD_RGB',phase=phase,images=number,new_inferred=new_inferred,source_cache_hits=cache_hits)
        write(directory/(phase+'_variants.json'),data_variants[phase])
    memory=gpu_end(record,'Qualified frozen PARSeq alone; U-Net and ByT5 not resident',
        c['runtime']['ocr_batch_size'],dict(actual_maximum_batch=max_batch,
            model_input=[max_batch,3,*model_input_hw],crop_shapes=sorted(map(list,input_shapes))),started)
    recognizer.release();del recognizer;gc.collect();torch.cuda.empty_cache()
    write(directory/'memory.json',memory);write(directory/'costs.json',dict(per_image=costs,
        actual_new_gpu_crop_inferences=new_inferred,source_cache_hits=cache_hits,
        note='Cold-start additional crops include source-cache hits not admitted by the original cap20. Generation/inference costs include rejected alternatives.'))
    # Real candidate utility uses spatial words recovered, never F1-only deletion.
    x,benefit,harm=[],[],[]
    for s in data['fit']:
        sid=s['sample_id'];words=annotations(s);base=predictions['fit'][sid]
        for variant in data_variants['fit'][sid]:
            after=[p for p in base if p['id']!=variant['parent']]+variant['children']
            labels=recovery_label(base,after,words,c['matching_iou'])
            x.append(variant['features']);benefit.append(labels['beneficial']);harm.append(int(labels['damaged']>0))
    beneficial,mb=fit_binary(x,benefit,c,'rereading actual recovery')
    damaging,mh=fit_binary(x,harm,c,'rereading damage')
    with (directory/'models.pkl').open('wb') as stream:pickle.dump(dict(benefit=beneficial,harm=damaging),stream)
    write(directory/'training.json',dict(benefit=mb,harm=mh))
    scores={}
    for phase in PHASES:
        scores[phase]={}
        for sid,variants in data_variants[phase].items():
            vectors=[v['features'] for v in variants]
            b,h=probability(beneficial,vectors),probability(damaging,vectors)
            scores[phase][sid]={v['id']:[b[i],h[i]] for i,v in enumerate(variants)}
    write(directory/'scores.json',scores)

    def predict(phase,bt,ht,confidence_only=False):
        output={}
        for sid,base in predictions[phase].items():
            kept=list(base);used=set()
            variants=data_variants[phase][sid]
            rank=lambda v: (-(v['confidence_gain'] if confidence_only else scores[phase][sid][v['id']][0]),v['id'])
            for variant in sorted(variants,key=rank):
                if variant['parent'] in used:continue
                if confidence_only:
                    if variant['confidence_gain']<=c['reread']['confidence_gain']:continue
                else:
                    b,h=scores[phase][sid][variant['id']]
                    if b<bt or h>ht:continue
                others=[p for p in kept if p['id']!=variant['parent']]
                if any(conflicts(child,q,c['compatibility_iou']) for child in variant['children'] for q in others):continue
                kept=others+variant['children'];used.add(variant['parent'])
            output[sid]=kept
        return output

    trials=[]
    for bt in c['reread']['benefit_thresholds']:
        for ht in c['reread']['harm_thresholds']:
            result=report(data['calibration'],predict('calibration',bt,ht),c,predictions['calibration'])
            trials.append(dict(benefit_threshold=bt,harm_threshold=ht,aggregate=result['aggregate']))
    valid=[r for r in trials if r['aggregate']['spatial']['damaged']<=c['reread']['damage_limit']]
    if not valid:raise RuntimeError('Reread KEEP policy missing')
    chosen=max(valid,key=lambda r:(r['aggregate']['spatial']['correct'],r['aggregate']['spatial']['f1'],r['benefit_threshold']))
    write(directory/'calibration.json',trials);write(directory/'selection_frozen.json',chosen)
    for phase in ('calibration','evaluation','test'):
        for name,pred in [('reread_selected',predict(phase,chosen['benefit_threshold'],chosen['harm_threshold'])),
                          ('reread_raw',predict(phase,c['reread']['raw_benefit_threshold'],c['reread']['raw_harm_threshold'])),
                          ('confidence_only',predict(phase,0,1,True))]:
            save_arm(directory,phase,name,data[phase],pred,c,predictions[phase])
    write(directory/'complete.json',dict(status='SUCCESS',selection=chosen))
    log(root,'REREAD_COMPLETE',selection=chosen,training=dict(benefit=mb,harm=mh),new_gpu_crops=new_inferred)


def comparison(candidate,baseline,metric,c):
    rows=candidate['per_image'];old={r['sample_id']:r for r in baseline['per_image']}
    if set(old)!={r['sample_id'] for r in rows}:raise ValueError('Unpaired comparison')
    a=np.asarray([[r[metric]['correct'],r[metric]['predictions'],r[metric]['gt_words'],
                   old[r['sample_id']][metric]['correct'],old[r['sample_id']][metric]['predictions']] for r in rows],float)
    rng=np.random.default_rng(c['seed']);draws=rng.integers(0,len(a),(c['bootstrap_draws'],len(a)))
    sums=a[draws].sum(1)
    delta=200*sums[:,0]/(sums[:,1]+sums[:,2])-200*sums[:,3]/(sums[:,4]+sums[:,2])
    return dict(delta_f1_points=100*(candidate['aggregate'][metric]['f1']-baseline['aggregate'][metric]['f1']),
        conditional_paired95_points=np.quantile(delta,[.025,.975]).tolist(),
        images_more_correct=int((a[:,0]>a[:,3]).sum()),images_fewer_correct=int((a[:,0]<a[:,3]).sum()),
        limitation='Fixed fitted models, historical TEST reuse; excludes search and training variability.')


def failure_breakdown(predictions,words,overlap):
    matched=set(exact_matching(predictions,words,overlap)['matched_prediction_indices'])
    kinds=Counter()
    for i,p in enumerate(predictions):
        if i in matched:continue
        compatible=[w for w in words if box_iou(p['bbox'],w['bbox'])>=overlap]
        if not p['text']:kind='empty_reading'
        elif any(w['text'].casefold()==p['text'].casefold() for w in compatible):kind='duplicate_exact_occurrence'
        elif compatible:kind='overlapping_word_wrong_transcription'
        else:kind='no_care_word_bbox_iou_0.5'
        kinds[kind]+=1
    return dict(kinds)


def final_report(root,c,data,smoke):
    for name in ('selectors','boundaries','reread'):
        if read(root/name/'complete.json')['status']!='SUCCESS':raise RuntimeError('Incomplete stage '+name)
    definitions={
        'baseline_raw':('', 'baseline_raw',None),
        'baseline_filter':('', 'baseline_filter','baseline_raw'),
        'additions':('selectors','additions','baseline_filter'),
        'triage_defer':('selectors','triage_defer','baseline_raw'),
        'triage_hard':('selectors','triage_hard','baseline_raw'),
        'reread_selected':('reread','reread_selected','triage_defer'),
        'reread_raw':('reread','reread_raw','triage_defer'),
        'confidence_only':('reread','confidence_only','triage_defer')}
    text_names=['baseline','byt5_previous','byt5_previous_boundaries_only','gap_head_raw','gap_head_selected']
    result=dict(status='SUCCESS',claim_label='approximate reproduction',
        result_class='smoke-test result' if smoke else 'approximate reproduction',smoke=smoke,
        protocol_sha256=sha(root/'protocol.json'),cohorts={},comparisons={},failure_analysis={},
        historical_test_reuse=True,independent_validation=False,
        metrics=dict(spatial='micro maximum one-to-one original bbox IoU>=0.5 and exact Unicode casefold; punctuation retained; all outputs',
                     lexical='micro per-image casefold whitespace-token multiset; punctuation retained; all outputs; no prediction ignore neutralization'),
        limitations=['Distinct candidate populations for budget20 selection versus all-original-block boundary correction; do not add gains.',
            'Annotation-overlap text labels are a care-GT proxy, not physical background/text labels.',
            'Previous MLP/GNN additions are frozen existing-run replays, not retrained in this battery.',
            'One seed; inherited calibration exposure, known RGB duplicates, and historically inspected TEST. No strict or Table5 claim.'])
    for phase in ('evaluation','test'):
        samples=data[phase];reference={s['sample_id']:s['baseline_raw'] for s in samples}
        for kind in ('raw','filter'):
            pred={s['sample_id']:s['baseline_'+kind] for s in samples}
            value=save_arm(root,phase,'baseline_'+kind,samples,pred,c,reference)
            if not smoke:
                expected=read(Path(c['paths']['filter_run'])/'selector'/(phase+'_results.json'))['baseline_budget20' if kind=='raw' else 'pointwise']['aggregate']
                if value['aggregate']['spatial']['correct']!=expected['correct'] or value['aggregate']['spatial']['predictions']!=expected['proposals']:
                    raise ValueError('Final paired baseline failed replay')
        reports={};predictions={}
        for name,(directory,file_name,control) in definitions.items():
            reports[name]=read(root/directory/f'{phase}_{file_name}_results.json')
            predictions[name]=read(root/directory/f'{phase}_{file_name}_predictions.json')
        # Independently executed prior models share the exact same frozen baseline.
        prior=Path(c['paths']['previous_additions_run'])/'additions'
        for kind in ('mlp','gnn'):
            original=read(prior/f'{phase}_{kind}_predictions.json')
            pred={s['sample_id']:[] for s in samples}
            with (root/'source_ocr.pkl').open('rb') as stream:old_ocr=pickle.load(stream)
            for sid in pred:
                pred[sid]=[dict(p,text=old_ocr[p['crop_array_sha256']]['raw_text']) for p in original[sid]]
            name='previous_additions_'+kind
            save_arm(root,phase,name,samples,pred,c,{s['sample_id']:s['baseline_filter'] for s in samples})
            reports[name]=read(root/f'{phase}_{name}_results.json');predictions[name]=pred
        result['cohorts'][phase]={k:r['aggregate'] for k,r in reports.items()}
        result['comparisons'][phase]={}
        for name,(_,_,control) in definitions.items():
            if control:
                result['comparisons'][phase][name]=dict(control=control,**{m:comparison(reports[name],reports[control],m,c) for m in ('spatial','lexical')})
        for kind in ('mlp','gnn'):
            name='previous_additions_'+kind
            result['comparisons'][phase][name]=dict(control='baseline_filter',**{m:comparison(reports[name],reports['baseline_filter'],m,c) for m in ('spatial','lexical')})
        text_reports={name:read(root/'boundaries'/f'{phase}_{name}_results.json') for name in text_names}
        result['cohorts'][phase]['text_boundary_population']={name:r['aggregate'] for name,r in text_reports.items()}
        for name in text_names[1:]:
            result['comparisons'][phase]['text/'+name]=dict(control='text/baseline',lexical=comparison(text_reports[name],text_reports['baseline'],'lexical',c))
        failures={name:Counter() for name in reports};changed=[]
        for s in samples:
            sid=s['sample_id'];words=annotations(s)
            for name in reports:
                failures[name].update(failure_breakdown(predictions[name][sid],words,c['matching_iou']))
            before=s['baseline_raw'];after=predictions['reread_selected'][sid]
            if [(p['bbox'],p['text']) for p in before]!=[(p['bbox'],p['text']) for p in after]:
                changed.append(dict(sample_id=sid,before=before,after=after,
                    changes=recovery_label(before,after,words,c['matching_iou'])))
        result['failure_analysis'][phase]=dict(false_positive_types={k:dict(v) for k,v in failures.items()},
            reread_changed_images=changed)
        if phase=='test':
            clean={name:dict(spatial=aggregate([r for r in value['per_image'] if r['sample_id'] not in c['known_duplicate_test_ids']],'spatial'),
                            lexical=aggregate([r for r in value['per_image'] if r['sample_id'] not in c['known_duplicate_test_ids']],'lexical'))
                   for name,value in reports.items()}
            result['without_known_duplicate_test_images']=clean
    result['memory']=dict(boundaries=read(root/'boundaries/memory.json'),reread=read(root/'reread/memory.json'))
    result['reread_costs']=read(root/'reread/costs.json')
    write(root/'results.json',result)
    lines=['# Selective word repair tests','', '**Approximate reproduction; historically explored cohorts.**','',
           'Budget20 track: exact casefold word plus original bbox IoU>=0.5. All care words and outputs count.','',
           '| TEST arm | Correct | Outputs | FP | FN | Precision | Recall | F1 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,v in result['cohorts']['test'].items():
        if name=='text_boundary_population':continue
        m=v['spatial'];lines.append(f"| {name} | {m['correct']} | {m['predictions']} | {m['fp']} | {m['fn']} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.2%} |")
    lines+=['','Original-block text-only boundary track: lexical matching without IoU. Do not merge its gains with the table above.','',
            '| TEST arm | Correct | Outputs | Recovered | Damaged | F1 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,value in result['cohorts']['test']['text_boundary_population'].items():
        m=value['lexical'];lines.append(f"| {name} | {m['correct']} | {m['predictions']} | {m['recovered']} | {m['damaged']} | {m['f1']:.2%} |")
    lines+=['','All policies use calibration only and include KEEP. Repeated TEST inspection, one seed and inherited calibration exposure limit generalization.',
            'Raw endpoints are descriptive controls, not TEST-selected replacements. Read results.json for paired conditional intervals, false-positive attribution, source provenance, costs and duplicate sensitivity.']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    log(root,'REPORT_COMPLETE',status='SUCCESS',result_sha256=sha(root/'results.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['approximate'])
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output-root',required=True,type=Path)
    parser.add_argument('--stage',required=True,choices=['prepare','selectors','boundaries','reread','report','all'])
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();root=args.output_root.resolve()
    if root.is_relative_to(REPO):parser.error('Use an explicit external output directory')
    try:
        c=checked_config(args.config,args.mode);runtime=setup(c)
        data=(prepare(root,c,runtime,args.smoke) if args.stage in ('prepare','all')
              else load_prepared(root,c,args.smoke))
        if args.stage in ('selectors','all'):selectors_stage(root,c,data)
        if args.stage in ('boundaries','all'):boundaries_stage(root,c,data,args.smoke)
        if args.stage in ('reread','all'):reread_stage(root,c,data)
        if args.stage in ('report','all'):final_report(root,c,data,args.smoke)
    except Exception as error:
        root.mkdir(parents=True,exist_ok=True)
        write(root/'FAILED.json',dict(status='FAILED',error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),oom=isinstance(error,torch.cuda.OutOfMemoryError)))
        raise


if __name__=='__main__':main()
