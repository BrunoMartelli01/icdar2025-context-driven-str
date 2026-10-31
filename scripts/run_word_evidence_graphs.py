"""Isolated APPROXIMATE typed/ownership/character/active-observation experiments."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from localize.word_graph_recovery import HypothesisScorer, preserve_and_add, conflicts
from localize.word_evidence_graphs import EvidenceGraph, relation_edges, visual_regions
from recognition.selective_word_repair import exact_matching
from scripts.run_selective_word_repair_tests import setup, gpu_start, gpu_end, report
from scripts.run_text_reconstruction_tests import read, write, sha, annotations

REPO=Path(__file__).resolve().parents[1]
PHASES=('fit','calibration','evaluation','test')


def log(root,stage,**values):
    write(root/'status.json',dict(status='SUCCESS' if stage=='COMPLETE' else 'RUNNING',stage=stage,
        pid=os.getpid(),**values))
    print(stage,json.dumps(values),flush=True)


def verify(path,expected):
    if sha(path)!=expected:raise ValueError('Scientific input hash differs: '+str(path))


def state_digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rgb_for(sample):
    verify(sample['image_path'],sample['image_sha256'])
    with Image.open(sample['image_path']) as image:return np.ascontiguousarray(image.convert('RGB'))


def matching_count(rows,words):return exact_matching(rows,words,.5)['correct']


def compatible_add(prefix,pool,scores,threshold,overlap):
    if threshold=='KEEP':return copy.deepcopy(prefix)
    return preserve_and_add(prefix,pool,scores,threshold=threshold,overlap=overlap)


def save_arm(root,name,phase,samples,predictions,c,prefix='baseline_filter'):
    reference={s['sample_id']:s[prefix] for s in samples}
    value=report(samples,predictions,c,reference)
    write(root/f'{phase}_{name}_predictions.json',predictions)
    write(root/f'{phase}_{name}_results.json',value)
    return value


def load(root):
    with (root/'prepared.pkl').open('rb') as stream:return pickle.load(stream)


def prepare(root,c,smoke):
    source=Path(c['paths']['graph_run']);selective=Path(c['paths']['selective_run'])
    for path,expected in c['input_sha256'].items():verify(path,expected)
    original=read(source/'resolved_config.json');protocol=read(source/'protocol.json')
    manifest=read(source/'artifact_manifest.json')['artifacts']
    def checked_graph(relative):
        path=source/relative;value=manifest[relative]
        verify(path,value['sha256'] if isinstance(value,dict) else value)
        return path
    samples=read(selective/'samples.json')
    selective_protocol=read(selective/'protocol.json')
    if sha(REPO/'recognition/parseq.py')!=selective_protocol['source_code_sha256']['recognition/parseq.py']:
        raise ValueError('Native PARSeq adapter changed from the qualified source')
    with (selective/'source_ocr.pkl').open('rb') as stream:ocr=pickle.load(stream)
    checkpoint=torch.load(checked_graph('additions/gnn/best.pt'),map_location='cpu',weights_only=True)
    base=HypothesisScorer(224,64,3,message_passing=True)
    base.load_state_dict(checkpoint['state_dict'],strict=True);base.eval()
    policy=read(checked_graph('additions/gnn/selection_frozen.json'))
    restored={e['sample_id']:e for e in read(original['paths']['input_restoration'])['restored']}
    data={};audits={};input_files={}
    for phase in PHASES:
        entries=samples[phase][:c['smoke_images'][phase]] if smoke else samples[phase]
        data[phase]=[]
        for old in entries:
            s=copy.deepcopy(old);sid=s['sample_id']
            if sha(s['image_path'])!=s['image_sha256']:
                if sid not in restored:raise ValueError('Unqualified changed RGB: '+sid)
                s['image_path']=restored[sid]['isolated_copy']
            verify(s['image_path'],s['image_sha256']);verify(s['annotation_path'],s['annotation_sha256'])
            verify(s['maps_path'],s['maps_sha256'])
            path=checked_graph('additions/data/'+phase+'/'+sid.replace(':','__')+'.npz')
            input_files[str(path)]=sha(path)
            with np.load(path) as z:arrays={k:z[k].copy() for k in z.files}
            if arrays['features'].shape!=(len(s['pool']),224):raise ValueError('Frozen feature order/size differs')
            for p in s['pool']:
                if p['text']!=ocr[p['crop_array_sha256']]['raw_text']:raise ValueError('Frozen reading differs')
            if len({p['crop_array_sha256'] for p in s['pool']})>c['base_crop_budget']:
                raise ValueError('Source OCR budget exceeds contract')
            with torch.inference_mode():
                scores=base((torch.from_numpy(arrays['features'])-checkpoint['mean'])/checkpoint['scale'],
                            torch.from_numpy(arrays['adjacency'])).sigmoid().tolist()
            s['baseline_gnn']=compatible_add(s['baseline_filter'],s['pool'],
                dict(zip([p['id'] for p in s['pool']],scores)),policy['threshold'],c['compatibility_iou'])
            support,conflict,collapsed=relation_edges(s['pool'],arrays['adjacency'],overlap=c['compatibility_iou'])
            with np.load(s['maps_path']) as z:
                regions,incidence=visual_regions(z['probabilities'],z['canvas'],s['pool'],s['baseline_filter'],**c['regions'])
            s['arrays']=dict(**arrays,support=support,conflict=conflict,collapsed=collapsed,regions=regions,incidence=incidence)
            s['words']=annotations(s) # FIT targets and separate evaluation only; never graph inputs.
            data[phase].append(s)
        if phase in ('evaluation','test'):
            archived=read(checked_graph('additions/'+phase+'_gnn_predictions.json'))
            for s in data[phase]:
                if [p['id'] for p in s['baseline_gnn']]!=[p['id'] for p in archived[s['sample_id']]]:
                    raise ValueError('Qualified GNN baseline does not replay: '+s['sample_id'])
        audits[phase]=dict(images=len(entries),candidates=sum(len(s['pool']) for s in data[phase]),
            visual_regions=sum(len(s['arrays']['regions']) for s in data[phase]))
        log(root,'PREPARE',split=phase,**audits[phase])
    with (root/'prepared.pkl').open('wb') as stream:pickle.dump(data,stream)
    with (root/'source_ocr.pkl').open('wb') as stream:pickle.dump(ocr,stream)
    write(root/'source_verification.json',dict(status='PASS',populations=audits,used_graph_arrays=input_files,
        graph_baseline_replay=True,prepared_sha256=sha(root/'prepared.pkl'),source_ocr_sha256=sha(root/'source_ocr.pkl')))
    for phase in ('calibration','evaluation','test'):
        for kind in ('filter','gnn'):
            save_arm(root,'baseline_'+kind,phase,data[phase],{s['sample_id']:s['baseline_'+kind] for s in data[phase]},c)
    if not smoke:
        value=read(root/'test_baseline_filter_results.json')['aggregate']['spatial']
        if [value['correct'],value['predictions'],value['gt_words']]!=[1187,1707,2204]:
            raise ValueError('Primary metric baseline differs')
        if read(root/'test_baseline_gnn_results.json')['aggregate']['spatial']['correct']!=1201:
            raise ValueError('GNN word matches differ')


def graph_stage(root,c,smoke):
    data=load(root);g=c['training'];directory=root/'graphs';directory.mkdir(exist_ok=False)
    vectors=np.concatenate([s['arrays']['features'] for s in data['fit'] if len(s['pool'])])
    mean=torch.tensor(vectors.mean(0),device='cuda');scale=torch.tensor(vectors.std(0),device='cuda').clamp_min(g['minimum_scale'])
    tensors={}
    for phase in PHASES:
        tensors[phase]=[]
        for s in data[phase]:
            a=s['arrays'];z={k:torch.tensor(a[k],device='cuda') for k in ('support','conflict','collapsed','regions','incidence')}
            z['x']=(torch.tensor(a['features'],device='cuda')-mean)/scale
            if phase=='fit':z.update(target=torch.tensor(a['target'],device='cuda'),valid=torch.tensor(a['valid'],device='cuda'))
            tensors[phase].append(z)
    def forward(model,z,region):
        return model(z['x'],z['support'],z['conflict'],z['collapsed'],
            z['regions'] if region else z['regions'][:0],z['incidence'] if region else z['incidence'][:,:0])
    for name,typed,region,ownership in [('collapsed',False,False,False),('typed',True,False,False),
            ('region_global',False,True,False),('ownership',False,True,True)]:
        branch=directory/name;branch.mkdir();torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
        memory=gpu_start(c,name);started=time.monotonic()
        model=EvidenceGraph(224,g['hidden'],g['layers'],typed=typed,ownership=ownership).cuda()
        initial=state_digest(model);opt=torch.optim.AdamW(model.parameters(),lr=g['learning_rate'],weight_decay=g['weight_decay'])
        rng=torch.Generator().manual_seed(c['seed']);history=[]
        initial_metric=read(root/'calibration_baseline_filter_results.json')['aggregate']['spatial']
        best=dict(epoch=0,threshold='KEEP',f1=initial_metric['f1'],predictions=initial_metric['predictions'])
        def predictions(phase,threshold):
            model.eval();output={}
            with torch.inference_mode():
                for s,z in zip(data[phase],tensors[phase]):
                    values=forward(model,z,region).sigmoid().cpu().tolist()
                    output[s['sample_id']]=compatible_add(s['baseline_filter'],s['pool'],
                        dict(zip([p['id'] for p in s['pool']],values)),threshold,c['compatibility_iou'])
            return output
        for epoch in range(1,(1 if smoke else g['epochs'])+1):
            losses=[];model.train()
            for j in torch.randperm(len(data['fit']),generator=rng).tolist():
                z=tensors['fit'][j]
                if not z['valid'].any():continue
                opt.zero_grad(set_to_none=True)
                loss=F.binary_cross_entropy_with_logits(forward(model,z,region)[z['valid']],z['target'][z['valid']])
                if not loss.isfinite():raise FloatingPointError('Nonfinite graph loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),g['gradient_clip'],error_if_nonfinite=True)
                opt.step();losses.append(float(loss.detach()))
            if not losses:raise ValueError('No graph FIT training updates')
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),updates=len(losses)))
            if epoch%g['calibration_every']==0 or smoke or epoch==g['epochs']:
                trials=[]
                for threshold in g['thresholds']:
                    value=report(data['calibration'],predictions('calibration',threshold),c)['aggregate']['spatial']
                    trials.append(dict(threshold=threshold,**value))
                    if (value['f1'],-value['predictions'])>(best['f1'],-best['predictions']):
                        best=dict(epoch=epoch,threshold=threshold,f1=value['f1'],predictions=value['predictions'])
                        torch.save(dict(state_dict=model.state_dict(),mean=mean.cpu(),scale=scale.cpu(),selection=best),branch/'best.pt')
                write(branch/f'epoch{epoch}_calibration.json',trials)
            write(branch/'history.json',history);log(root,'GRAPH_TRAIN',arm=name,**history[-1],best=best)
        final_digest=state_digest(model)
        if final_digest==initial:raise ValueError('Graph parameters did not update')
        torch.save(dict(state_dict=model.state_dict(),mean=mean.cpu(),scale=scale.cpu()),branch/'last.pt')
        for phase in ('evaluation','test'):
            save_arm(root,name+'_final',phase,data[phase],predictions(phase,g['raw_threshold']),c)
        if best['epoch']:model.load_state_dict(torch.load(branch/'best.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
        for phase in ('evaluation','test'):
            save_arm(root,name,phase,data[phase],predictions(phase,best['threshold']),c)
        write(branch/'selection_frozen.json',best)
        write(branch/'qualification.json',dict(parameters=sum(p.numel() for p in model.parameters()),
            initial_parameter_sha256=initial,final_parameter_sha256=final_digest,
            memory=gpu_end(memory,'EvidenceGraph + AdamW; frozen cached U-Net/PARSeq evidence',1,[20,224],started)))
        del model,opt;torch.cuda.empty_cache()


def run(args):
    c=read(args.config);root=args.output.resolve()
    if (args.mode!='approximate' or REPO/'configs/approximate' not in args.config.resolve().parents
            or c['reproduction_mode']!='approximate' or c['experiment_kind']!='word_evidence_graphs_v1'):
        raise ValueError('Explicit isolated APPROXIMATE configuration required')
    if args.stage=='all' or args.stage=='prepare':
        root.mkdir(exist_ok=False,parents=True)
        write(root/'resolved_config.json',c)
        runtime=setup(c)
        code={str(p.relative_to(REPO)):sha(p) for folder in ('localize','recognition','scripts') for p in (REPO/folder).glob('*.py')}
        write(root/'protocol.json',dict(config_sha256=sha(args.config),code_sha256=code,runtime=runtime,smoke=args.smoke,
            mode='approximate',result_class='smoke-test result' if args.smoke else 'approximate reproduction',
            metric='maximum one-to-one bbox IoU>=.5 AND exact Unicode casefold; punctuation retained; all care GT and predictions count',
            inherited_calibration_exposure=True,historical_test_exploration=True,independent_validation=False,seed_count=1))
        prepare(root,c,args.smoke)
    else:
        setup(c)
        verify(root/'resolved_config.json',sha(args.config))
    stages={'graphs':graph_stage}
    from scripts.word_evidence_consensus import alignment_stage,consensus_stage
    from scripts.word_evidence_active import active_stage
    from scripts.analyze_word_evidence_graphs import analyze
    stages.update(alignment=alignment_stage,consensus=consensus_stage,active=active_stage,report=analyze)
    for name,method in stages.items():
        if args.stage in ('all',name):method(root,c,args.smoke)
    if args.stage in ('all','report'):log(root,'COMPLETE')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['approximate'],required=True)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['all','prepare','graphs','alignment','consensus','active','report'],default='all')
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    try:run(args)
    except Exception as error:
        write(args.output/'failure.json',dict(status='FAILED',error=str(error),traceback=traceback.format_exc(),
            oom=isinstance(error,torch.cuda.OutOfMemoryError)))
        raise
