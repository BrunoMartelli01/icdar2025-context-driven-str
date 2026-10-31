"""Four controlled APPROXIMATE follow-ups; all choices use calibration word F1."""
from __future__ import annotations
import argparse
import copy
from functools import lru_cache
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from localize.graph_occlusion_followups import (predicted_strip_mask,apply_native_occlusion,occlude_original_rgb,
    positive_edges,use_enriched_graph,rectangle_points,resample_boundary,ContourPointRefiner,
    cyclic_contour_loss,contour_boxes)
from localize.graph_occlusion_research import (FragmentGraph,fragment_labels,neighbor_graph,
    graph_supervision,balanced_binary_loss,occlude_instances,teacher_reliability,
    pixel_distillation,instance_structure_distillation)
from localize.occlusion_four_tests import decode_instance_labels,row,features,iou
from scripts.run_graph_occlusion_research import (budget_and_filter,baseline_filter,dataset_for,
    graph_records,recognized_segmentation,record_failure)
from scripts.run_teacher_student_tests import (source_model,source_records,supervised_loss,release,peaks,report)
from scripts.run_occlusion_four_tests import load_rgb,path_for,labels_for,changes
from scripts.run_mask_crop_budget import read,write,utc,setup,CropOCR
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _tensor
from scripts.evaluate_cached_totaltext_padding import crop_hash
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.data import file_sha256
from segmentation.training import seed_everything
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask,remap_letterbox_native_box

REPO=Path(__file__).resolve().parents[1]


def announce(out,phase,**values):
    write(out/'status.json',dict(status='RUNNING',phase=phase,utc=utc(),pid=os.getpid(),**values))
    print(phase,json.dumps(values),flush=True)


@lru_cache(maxsize=2)
def prior_manifest(path):return read(Path(path))['artifacts']


def previous(c,name):
    path=Path(c['prior_run'])/name
    if file_sha256(path)!=prior_manifest(c['paths']['prior_artifact_manifest'])[name]['sha256']:
        raise ValueError('Prior artifact changed: '+name)
    return path


def prepare(args):
    cp=args.config.resolve();c=read(cp)
    if (args.mode!='approximate' or REPO/'configs/approximate' not in cp.parents
            or c['reproduction_mode']!='approximate' or c['experiment_kind']!='graph_occlusion_followups_v2'):
        raise ValueError('Explicit isolated APPROXIMATE configuration required')
    for name,digest in c['input_sha256'].items():
        if file_sha256(Path(c['paths'][name]))!=digest:raise ValueError('Input drift: '+name)
    original=read(Path(c['source_run'])/'protocol.json')
    if file_sha256(Path(c['source_run'])/'results.json')!=c['source_results_sha256']:
        raise ValueError('Original source results changed')
    old=read(previous(c,'protocol.json'))
    if read(previous(c,'results.json'))['status']!='SUCCESS' or read(previous(c,'independent_audit.json'))['status']!='PASS':
        raise ValueError('Successful independently audited prior required')
    for name in c['frozen_source_files']:
        if file_sha256(REPO/name)!=original['source_code_sha256'][name]:raise ValueError('Frozen original changed: '+name)
    for name in c['followup_frozen_files']:
        if file_sha256(REPO/name)!=old['identity']['code_sha256'][name]:raise ValueError('Qualified prior implementation changed: '+name)
    for name,digest in c['filter_source_sha256'].items():
        if file_sha256(Path(c['filter_source'])/name)!=digest:raise ValueError('Filter artifact changed: '+name)
    if c['metric_iou_threshold']!=.5:raise ValueError('Qualified IoU0.5 metric required')
    index=copy.deepcopy(old['identity']['index'])
    if [len(index[k]) for k in ('fit','calibration','evaluation','test')]!=[1030,100,125,300]:
        raise ValueError('Qualified populations changed')
    if args.smoke:index={k:v[:c['smoke'][k]] for k,v in index.items()}
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    identity=dict(config_sha256=file_sha256(cp),index=index,smoke=args.smoke,
        code_sha256={str(p.relative_to(REPO)):file_sha256(p) for folder in ('localize','scripts','models','segmentation','recognition') for p in sorted((REPO/folder).rglob('*.py'))})
    hardware=setup(c)
    if (out/'protocol.json').exists():
        if read(out/'protocol.json')['identity']!=identity:raise ValueError('Resume identity changed')
    else:
        write(out/'protocol.json',dict(identity=identity,resolved_config=c,hardware=hardware,metric=old['metric'],
            result_class='smoke-test result' if args.smoke else 'approximate reproduction; exploratory component experiments',
            selection_uses_test=False,historical_test_exploration=True,seed_count=1))
        write(out/'resolved_config.json',c)
    return c,out,index,read(c['paths']['source_contract'])


def source(c,index):
    records,readings=source_records(c,index)
    for phase,entries in index.items():
        for entry in entries:records[phase][entry['sample_id']].update(entry)
    for name in ('graph/ocr.jsonl','segmentation/ocr.jsonl'):
        for line in previous(c,name).open():
            r=json.loads(line);key=r['crop_array_sha256']
            if key in readings and readings[key]['raw_text']!=r['raw_text']:raise ValueError('Frozen OCR transcript conflict')
            readings[key]=r
    return records,readings


def old_structure(c,phase,entries):
    name='segmentation/structure/epoch6_calibration/records.json' if phase=='calibration' else 'segmentation/structure/'+phase+'/records.json'
    values=read(previous(c,name))
    return {e['sample_id']:values[e['sample_id']] for e in entries}


def save_endpoint(directory,name,phase,records,predictions,ocr):
    write(directory/(phase+'_'+name+'_predictions.json'),predictions)
    result=report(records,predictions,ocr.rows)
    write(directory/(phase+'_'+name+'_results.json'),result)
    return result


def threshold_stage(c,out,index,contract,smoke):
    directory=out/'threshold';directory.mkdir(exist_ok=False)
    records,readings=source(c,index);ocr=CropOCR(c,directory/'ocr.jsonl');ocr.rows.update(readings)
    policies={};calibration={}
    for name in ('baseline','structure'):
        cal=records['calibration'] if name=='baseline' else old_structure(c,'calibration',index['calibration'])
        trials=[]
        for threshold in c['recalibration']['thresholds']:
            tuned=copy.deepcopy(c);tuned['filter']['threshold']=threshold
            predictions=budget_and_filter(tuned,cal,ocr)
            trials.append(dict(threshold=threshold,aggregate=report(cal,predictions,ocr.rows)['aggregate']))
        choice=max(range(len(trials)),key=lambda j:(trials[j]['aggregate']['f1'],-j))
        policies[name]=trials[choice]['threshold'];calibration[name]=trials
    write(directory/'selection_frozen.json',policies);write(directory/'calibration.json',calibration)
    for phase in ('evaluation','test'):
        for name in ('baseline','structure'):
            values=records[phase] if name=='baseline' else old_structure(c,phase,index[phase])
            tuned=copy.deepcopy(c);tuned['filter']['threshold']=policies[name]
            predictions=budget_and_filter(tuned,values,ocr)
            result=save_endpoint(directory,'recal_'+name,phase,values,predictions,ocr)
            if name=='baseline':
                ref=baseline_filter(c,records[phase],phase,ocr)
                save_endpoint(directory,'baseline_filter',phase,records[phase],ref,ocr)
            else:
                fixed=budget_and_filter(c,values,ocr)
                save_endpoint(directory,'source_structure',phase,values,fixed,ocr)
            announce(out,'THRESHOLD_RESULT',arm=name,split=phase,threshold=policies[name],f1=result['aggregate']['f1'])
    ocr.release();write(directory/'complete.json',dict(status='SUCCESS',policies=policies,scope='Frozen model/filter weights; calibration-only thresholds'))


def continuation_stage(c,out,index,contract,smoke):
    directory=out/'continuation';directory.mkdir(exist_ok=False)
    ds,by=dataset_for(c,contract);batches=[ds[by[e['sample_id']]] for e in index['fit']]
    records,readings=source(c,index);ocr=CropOCR(c,directory/'ocr.jsonl');ocr.rows.update(readings)
    config=c['segmentation'];warm=torch.load(c['paths']['structure_checkpoint'],map_location='cpu',weights_only=True)
    if warm['selection']['epoch']!=6:raise ValueError('Qualified epoch6 checkpoint required')
    initial_cal=old_structure(c,'calibration',index['calibration'])
    initial_pred=budget_and_filter(c,initial_cal,ocr);initial_f1=report(initial_cal,initial_pred,ocr.rows)['aggregate']['f1']
    audits={};selections={}
    for arm,lr in c['continuation']['learning_rates'].items():
        seed_everything(c['seed'],deterministic_algorithms=True)
        student=source_model(c,contract);student.load_state_dict(warm['state_dict'],strict=True)
        teacher=source_model(c,contract).requires_grad_(False).eval()
        original={k:v.cpu().clone() for k,v in teacher.state_dict().items()}
        optimizer=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=lr,weight_decay=config['weight_decay'])
        generator=torch.Generator().manual_seed(c['seed']);branch=directory/arm;branch.mkdir()
        best=dict(additional_epoch=0,source_epoch=6,f1=initial_f1);history=[];torch.cuda.reset_peak_memory_stats()
        epochs=1 if smoke else c['continuation']['additional_epochs']
        for epoch in range(1,epochs+1):
            student.train();order=torch.randperm(len(batches),generator=generator).tolist();losses=[]
            for start in range(0,len(order),config['batch_size']):
                chosen=[batches[j] for j in order[start:start+config['batch_size']]]
                batch={k:torch.stack([b[k] for b in chosen]).cuda() for k in ('image','mask','kernel','validity','separation','owners_native')}
                flips=torch.rand(len(chosen),generator=generator)<.5
                for k in batch:batch[k][flips]=batch[k][flips].flip(-1)
                weak=batch['image'];contrast=1+float(torch.rand((),generator=generator)*2-1)*config['contrast']
                brightness=float(torch.rand((),generator=generator)*2-1)*config['brightness']
                strong,_=occlude_instances(weak*contrast+brightness,batch['owners_native'],generator,
                    probability=config['occlusion_probability'],width_fraction=config['occlusion_width_fraction'],maximum_words=config['maximum_occluded_words'])
                optimizer.zero_grad(set_to_none=True);logits=student(strong)
                with torch.no_grad():target=teacher(weak).sigmoid()
                reliable=teacher_reliability(target,torch.cat((batch['mask'],batch['kernel']),1),batch['validity'],config['teacher_confidence'])
                loss=supervised_loss(logits,batch,contract['resolved_config']['loss'])
                loss=loss+config['pixel_weight']*pixel_distillation(logits,target,reliable)
                loss=loss+config['structure_weight']*instance_structure_distillation(logits,target,reliable,batch['owners_native'],batch['separation'])
                if not loss.isfinite():raise FloatingPointError('Nonfinite continuation loss')
                loss.backward();torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],config['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();losses.append(float(loss.detach()))
            history.append(dict(additional_epoch=epoch,loss=float(np.mean(losses)),updates=len(losses),learning_rate=lr))
            write(branch/'history.json',history);optimizer.zero_grad(set_to_none=True)
            torch.save(dict(state_dict=student.state_dict(),additional_epoch=epoch),branch/'last.pt')
            announce(out,'CONTINUATION_TRAIN',arm=arm,**history[-1])
            if epoch%c['continuation']['calibration_every']==0 or epoch==epochs:
                student.eval();_,_,metric=recognized_segmentation(student,c,contract,index['calibration'],branch/f'epoch{epoch}_calibration',ocr)
                write(branch/f'epoch{epoch}_calibration/results.json',metric)
                if metric['aggregate']['f1']>best['f1']:
                    best=dict(additional_epoch=epoch,source_epoch=6,f1=metric['aggregate']['f1'])
                    torch.save(dict(state_dict=student.state_dict(),selection=best),branch/'best.pt')
                announce(out,'CONTINUATION_CALIBRATION',arm=arm,epoch=epoch,f1=metric['aggregate']['f1'],best=best);ocr.release()
        changed=[k for k,v in student.state_dict().items() if k.startswith('encoder.') and not torch.equal(v.cpu(),original[k])]
        teacher_changed=[k for k,v in teacher.state_dict().items() if not torch.equal(v.cpu(),original[k])]
        updated=[k for k,v in student.state_dict().items() if not torch.equal(v.cpu(),warm['state_dict'][k])]
        if changed or teacher_changed or not updated:raise ValueError('Frozen encoder/teacher or student-update contract failed')
        audits[arm]=dict(encoder_changed=changed,teacher_changed=teacher_changed,student_changed_tensors=len(updated))
        write(branch/'memory.json',peaks('epoch6 U-Net student + original clean teacher + AdamW; PARSeq at calibration',config['batch_size'],[config['batch_size'],3,512,512]))
        write(branch/'selection_frozen.json',best);selections[arm]=best
        del student,teacher,optimizer,original;release()
        for name,checkpoint in ((arm,branch/'best.pt' if best['additional_epoch'] else Path(c['paths']['structure_checkpoint'])),(arm+'_final',branch/'last.pt')):
            model=source_model(c,contract);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'],strict=True)
            for phase in ('evaluation','test'):
                exported,predictions,_=recognized_segmentation(model,c,contract,index[phase],branch/name/phase,ocr)
                save_endpoint(directory,name,phase,exported,predictions,ocr)
            del model;ocr.release();release()
    write(directory/'weight_audit.json',dict(status='PASS',arms=audits))
    write(directory/'complete.json',dict(status='SUCCESS',selections=selections,optimizer_restart='Fresh matched AdamW in both arms; source checkpoint has no optimizer/RNG state'))


def graph_arrays(prob,deep,rgb,c,contract):
    g=c['graph'];pp=contract['resolved_config']['postprocessing']
    canvas=np.asarray(letterbox_canvas_mask((rgb.shape[1],rgb.shape[0]),native_size=prob.shape[1]),bool)
    labels=decode_instance_labels(prob,None,canvas,pp,None,'cut')
    atoms,audit=fragment_labels(prob,labels,threshold=g['seed_threshold'],minimum_seed_area=g['minimum_seed_area'],
        maximum_pieces=g['maximum_pieces'],maximum_nodes=g['maximum_nodes'])
    boxes=[];vectors=[];candidates=[]
    for identity in range(1,int(atoms.max())+1):
        yy,xx=np.where(atoms==identity);native=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]
        box=remap_letterbox_native_box(native,rgb.shape[:2][::-1],native_size=atoms.shape[0])
        p=row(box,rgb.shape[:2],'atom:'+str(identity),'baseline');p['bbox_native']=native
        value=features(p,rgb,prob[0],text_threshold=pp['text_threshold']);pooled=deep[:,yy,xx]
        vectors.append(value+pooled.mean(1).tolist()+pooled.std(1).tolist());boxes.append(box);candidates.append(p)
    dimensions=len(features(row([0,0,1,1],rgb.shape[:2],'dummy','baseline'),rgb,prob[0],text_threshold=pp['text_threshold']))+deep.shape[0]*2
    adjacency,edges=neighbor_graph(boxes,neighbors=g['neighbors'],maximum_distance=g['maximum_distance'])
    return dict(features=np.asarray(vectors,np.float32).reshape(len(vectors),dimensions),adjacency=adjacency,
        edges=edges,atoms=atoms,probabilities=prob),dict(boxes=boxes,candidates=candidates,audit=audit)


def original_graph(c,phase,sid):
    name='graph_data/'+phase+'/'+sid.replace(':','__')
    metadata=read(previous(c,name+'.json'));path=previous(c,name+'.npz')
    with np.load(path) as z:arrays={k:z[k].copy() for k in z.files}
    return metadata,arrays


def fragment_data_stage(c,out,index,contract,smoke):
    directory=out/'fragment_data';directory.mkdir(exist_ok=False)
    ds,by=dataset_for(c,contract);model=source_model(c,contract).requires_grad_(False)
    transform=_transform(contract['resolved_preprocessing'],'letterbox',train=False)
    generator=torch.Generator().manual_seed(c['seed']);counts=dict(clean_positive_edges=0,augmented_positive_edges=0,images_with_more_positive_edges=0,images=0)
    torch.cuda.reset_peak_memory_stats()
    for entry in index['fit']:
        sid=entry['sample_id'];clean_meta,clean=original_graph(c,'fit',sid);rgb=load_rgb(entry)
        cached=read(path_for(Path(c['source_run']),'fit',sid));boxes=[p['bbox_native'] for p in cached['arms']['mk']['baseline']]
        cut=predicted_strip_mask(boxes,clean['atoms'].shape,generator,fraction=c['enrichment']['width_fraction'],maximum_words=c['enrichment']['maximum_words'])
        visible_rgb=occlude_original_rgb(rgb,cut,c['enrichment']['fill_rgb'])
        tensor=_tensor(visible_rgb,transform).cuda()
        with torch.no_grad():evidence=model.forward_evidence(tensor)
        raw=evidence['words'][0].sigmoid().cpu().numpy();prob=raw.copy();prob[:,cut]=0
        deep=evidence['features'][0].cpu().numpy();arrays,metadata=graph_arrays(prob,deep,visible_rgb,c,contract)
        batch=ds[by[sid]];assignments,eligible=graph_supervision(arrays['atoms'],batch['owners_native'].numpy(),batch['validity'][0].numpy(),purity=c['graph']['owner_purity'])
        arrays.update(assignments=assignments,eligible=eligible,raw_probabilities=raw,cut_mask=cut)
        name=sid.replace(':','__');target=directory/(name+'.npz');np.savez_compressed(target,**arrays)
        write(directory/(name+'.json'),dict(**entry,**metadata,arrays_sha256=file_sha256(target),
            clean_positive_edges=positive_edges(clean),augmented_positive_edges=positive_edges(arrays)))
        counts['clean_positive_edges']+=positive_edges(clean);counts['augmented_positive_edges']+=positive_edges(arrays)
        counts['images_with_more_positive_edges']+=int(positive_edges(arrays)>positive_edges(clean));counts['images']+=1
        if counts['images']%50==0:announce(out,'FRAGMENT_DATA',**counts)
    write(directory/'memory.json',peaks('Frozen U-Net on predicted-box RGB occlusions; artificial strip removed from TRAIN proposal maps',1,[1,3,512,512]))
    if not smoke and counts['augmented_positive_edges']<counts['clean_positive_edges']*c['enrichment']['minimum_positive_multiplier']:
        raise ValueError('Insufficient extra positive links; inspect augmentation before training')
    write(directory/'complete.json',dict(status='SUCCESS',**counts));del model,evidence;release()


def graph_payload(c,out,phase,sid,augmented=False):
    if not augmented:return original_graph(c,phase,sid)
    path=out/'fragment_data'/sid.replace(':','__');metadata=read(path.with_suffix('.json'))
    if file_sha256(path.with_suffix('.npz'))!=metadata['arrays_sha256']:raise ValueError('Enriched graph cache changed')
    with np.load(path.with_suffix('.npz')) as z:arrays={k:z[k].copy() for k in z.files}
    return metadata,arrays


def validate_fragment(model,mean,scale,c,out,entries,phase,trace,ocr):
    # Reuse the original, byte-verified clean proposal graphs at inference.
    for entry in entries:previous(c,'graph_data/'+phase+'/'+entry['sample_id'].replace(':','__')+'.json')
    records=graph_records(model,mean,scale,c,Path(c['prior_run']),entries,phase,c['enrichment']['inference_policy'],trace)
    predictions=budget_and_filter(c,records,ocr)
    return records,predictions,report(records,predictions,ocr.rows)


def fragment_graph_stage(c,out,index,contract,smoke):
    directory=out/'fragment_graph';directory.mkdir(exist_ok=False);g=c['graph']
    clean=[];augmented=[]
    for entry in index['fit']:
        _,a=graph_payload(c,out,'fit',entry['sample_id']);_,b=graph_payload(c,out,'fit',entry['sample_id'],True)
        clean.append(a);augmented.append(b)
    vectors=torch.from_numpy(np.concatenate([z['features'] for z in clean if len(z['features'])]))
    mean=vectors.mean(0).cuda();scale=vectors.std(0).clamp_min(g['minimum_feature_scale']).cuda()
    def pack(z):
        result={k:torch.from_numpy(z[k]).cuda() for k in ('features','adjacency','edges','assignments','eligible')}
        result['features']=(result['features']-mean)/scale;return result
    base=[pack(z) for z in clean];extra=[pack(z) for z in augmented]
    records,readings=source(c,index);ocr=CropOCR(c,directory/'ocr.jsonl');ocr.rows.update(readings)
    baseline=baseline_filter(c,records['calibration'],'calibration',ocr)
    initial_f1=report(records['calibration'],baseline,ocr.rows)['aggregate']['f1'];selections={};capacities={}
    for arm in ('self_standard','gnn_standard','self_enriched','gnn_enriched'):
        seed_everything(c['seed'],deterministic_algorithms=True)
        model=FragmentGraph(vectors.shape[1],g['hidden'],g['layers'],arm.startswith('gnn')).cuda()
        capacities[arm]=sum(p.numel() for p in model.parameters())
        optimizer=torch.optim.AdamW(model.parameters(),lr=g['learning_rate'],weight_decay=g['weight_decay'])
        order_rng=torch.Generator().manual_seed(c['seed']);variant_rng=torch.Generator().manual_seed(c['seed']+1)
        branch=directory/arm;branch.mkdir();history=[];best=dict(epoch=0,f1=initial_f1,identity=True)
        torch.cuda.reset_peak_memory_stats();epochs=1 if smoke else g['epochs']
        for epoch in range(1,epochs+1):
            model.train();losses=[];selected_augmented=0
            for j in torch.randperm(len(base),generator=order_rng).tolist():
                choose=arm.endswith('enriched') and use_enriched_graph(clean[j],augmented[j],variant_rng,c['enrichment']['probability'])
                z=extra[j] if choose else base[j];selected_augmented+=int(choose)
                if not len(z['features']):continue
                optimizer.zero_grad(set_to_none=True);node,edge=model(z['features'],z['adjacency'],z['edges'])
                a=z['assignments'];v=z['eligible'];e=z['edges']
                same=(a[e[:,0]]==a[e[:,1]])&(a[e[:,0]]>0);valid=v[e[:,0]]&v[e[:,1]]
                loss=balanced_binary_loss(node,(a>0).float(),v)+g['edge_loss_weight']*balanced_binary_loss(edge,same.float(),valid)
                if not loss.isfinite():raise FloatingPointError('Nonfinite enriched graph loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),g['gradient_clip'],error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),updates=len(losses),augmented_images=selected_augmented))
            write(branch/'history.json',history);announce(out,'FRAGMENT_TRAIN',arm=arm,**history[-1])
            torch.save(dict(state_dict=model.state_dict(),mean=mean.cpu(),scale=scale.cpu(),epoch=epoch),branch/'last.pt')
            if epoch%c['enrichment']['calibration_every']==0 or epoch==epochs:
                _,_,metric=validate_fragment(model,mean,scale,c,out,index['calibration'],'calibration',branch/f'epoch{epoch}_calibration',ocr)
                write(branch/f'epoch{epoch}_calibration/results.json',metric)
                if metric['aggregate']['f1']>best['f1']:
                    best=dict(epoch=epoch,f1=metric['aggregate']['f1'],identity=False)
                    torch.save(dict(state_dict=model.state_dict(),selection=best),branch/'best.pt')
                announce(out,'FRAGMENT_CALIBRATION',arm=arm,epoch=epoch,f1=metric['aggregate']['f1'],best=best);ocr.release()
        selections[arm]=best;write(branch/'selection_frozen.json',best)
        write(branch/'memory.json',peaks('Fragment graph + AdamW + cached paired graphs; frozen PARSeq at calibration',1,['variable_nodes',vectors.shape[1]]))
        for name,checkpoint in ((arm,branch/'best.pt' if not best['identity'] else None),(arm+'_final',branch/'last.pt')):
            if checkpoint is not None:model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'],strict=True)
            for phase in ('evaluation','test'):
                if checkpoint is None:
                    exported=records[phase];predictions=baseline_filter(c,exported,phase,ocr)
                else:exported,predictions,_=validate_fragment(model,mean,scale,c,out,index[phase],phase,branch/name/phase,ocr)
                save_endpoint(directory,name,phase,exported,predictions,ocr)
        del model,optimizer;ocr.release();release()
    if len(set(capacities.values()))!=1:raise ValueError('Graph controls have different capacity')
    write(directory/'complete.json',dict(status='SUCCESS',selections=selections,parameters=capacities))


def paired_comparison(result,reference,c,rng):
    before={r['sample_id']:r for r in reference['per_image']};array=[];detail=[]
    for row in result['per_image']:
        b=before[row['sample_id']];n=row['gt_words']+row['proposals'];d=b['gt_words']+b['proposals']
        detail.append(dict(sample_id=row['sample_id'],delta_f1_points=(200*row['correct']/n if n else 0)-(200*b['correct']/d if d else 0),
            recovered_gt_indices=sorted(set(row['correct_gt_indices'])-set(b['correct_gt_indices'])),
            damaged_gt_indices=sorted(set(b['correct_gt_indices'])-set(row['correct_gt_indices']))))
        array.append([row['correct'],row['proposals'],row['gt_words'],b['correct'],b['proposals']])
    array=np.asarray(array,float);draws=rng.integers(0,len(array),(c['bootstrap_draws'],len(array)));sums=array[draws].sum(1)
    values=200*sums[:,0]/(sums[:,1]+sums[:,2])-200*sums[:,3]/(sums[:,4]+sums[:,2])
    return dict(delta_f1_points=100*(result['aggregate']['f1']-reference['aggregate']['f1']),**changes(result,reference),
        conditional_paired_95_percentile_interval=np.quantile(values,[.025,.975]).tolist(),per_image=detail)


def report_stage(c,out,index,contract,smoke):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    records,readings=source(c,index)
    for directory in ('threshold','continuation','fragment_graph','contour_graph'):
        if read(out/directory/'complete.json')['status']!='SUCCESS':raise ValueError('Incomplete required stage: '+directory)
        for line in (out/directory/'ocr.jsonl').open():
            r=json.loads(line);key=r['crop_array_sha256']
            if key in readings and readings[key]['raw_text']!=r['raw_text']:raise ValueError('OCR transcript conflict')
            readings[key]=r
    controls=dict(source_structure='baseline_filter',recal_baseline='baseline_filter',recal_structure='recal_baseline',
        continuation_control='source_structure',continuation_low_lr='continuation_control',
        graph_self_standard='baseline_filter',graph_gnn_standard='graph_self_standard',
        graph_self_enriched='graph_self_standard',graph_gnn_enriched='graph_gnn_standard',
        contour_self='baseline_filter',contour_gnn='contour_self')
    for arm,control in list(controls.items()):
        if arm.startswith(('continuation_','graph_','contour_')):
            controls[arm+'_final']=control+'_final' if control.startswith(('continuation_','graph_','contour_')) else control
    sections={'threshold':{'baseline_filter':'baseline_filter','source_structure':'source_structure','recal_baseline':'recal_baseline','recal_structure':'recal_structure'},
        'continuation':{name:'continuation_'+name for arm in c['continuation']['learning_rates'] for name in (arm,arm+'_final')},
        'fragment_graph':{name:'graph_'+name for arm in ('self_standard','gnn_standard','self_enriched','gnn_enriched') for name in (arm,arm+'_final')},
        'contour_graph':{name:'contour_'+name for arm in ('self','gnn') for name in (arm,arm+'_final')}}
    audited=0;summary={};md=['# Four controlled graph/occlusion follow-ups','',
        '**Smoke-test result.**' if smoke else '**Approximate reproduction; exploratory component experiments.**','',
        'Micro word F1: exact Unicode casefold retaining punctuation AND unpadded bbox IoU>=0.5. All care GT and outputs count. Historical TEST reuse, source calibration exposure and one seed limit interpretation.','']
    for phase in ('evaluation','test'):
        results={};predictions={}
        for directory,names in sections.items():
            for name,label in names.items():
                results[label]=read(out/directory/(phase+'_'+name+'_results.json'))
                predictions[label]=read(out/directory/(phase+'_'+name+'_predictions.json'))
        gt,hashes=labels_for(records[phase]);rng=np.random.default_rng(c['seed']);comparisons={}
        for arm,result in results.items():
            by={r['sample_id']:r for r in result['per_image']}
            if set(by)!=set(gt) or set(predictions[arm])!=set(gt):raise ValueError('Changed evaluation population')
            for sid,pp in predictions[arm].items():
                matrix=np.array([[iou(p['bbox'],w['bbox'])>=.5 and readings[p['crop_array_sha256']]['raw_text'].casefold()==w['text'].casefold() for w in gt[sid]] for p in pp],bool).reshape(len(pp),len(gt[sid]))
                correct=int((maximum_bipartite_matching(csr_matrix(matrix),perm_type='column')>=0).sum()) if matrix.size else 0
                if correct!=by[sid]['correct'] or len(pp)!=by[sid]['proposals'] or len(gt[sid])!=by[sid]['gt_words']:
                    raise ValueError('Independent word matching failed: '+arm+' '+sid)
                audited+=1
            if arm in controls:
                comparisons[arm]=dict(control=controls[arm],matched=paired_comparison(result,results[controls[arm]],c,rng),
                    versus_baseline=paired_comparison(result,results['baseline_filter'],c,rng))
        md += [f'## {phase}','','| Arm | F1 % | Delta vs baseline | Matched control | Delta vs control |','| --- | ---: | ---: | --- | ---: |']
        for arm,result in results.items():
            value=comparisons.get(arm)
            md.append(f"| {arm} | {result['aggregate']['f1']*100:.3f} | {value['versus_baseline']['delta_f1_points']:+.3f} | {value['control']} | {value['matched']['delta_f1_points']:+.3f} |" if value else f"| {arm} | {result['aggregate']['f1']*100:.3f} | — | — | — |")
        write(out/(phase+'_results.json'),results);write(out/(phase+'_predictions.json'),predictions)
        write(out/(phase+'_comparisons.json'),comparisons);write(out/(phase+'_annotation_hashes.json'),hashes)
        summary[phase]={k:v['aggregate'] for k,v in results.items()}
    write(out/'independent_audit.json',dict(status='PASS',image_arm_checks=audited,algorithm='Independent SciPy maximum bipartite matching'))
    (out/'comparison.md').write_text('\n'.join(md)+'\n')
    write(out/'results.json',dict(status='SUCCESS',claim_label='approximate reproduction',smoke=smoke,results=summary))
    write(out/'status.json',dict(status='SUCCESS',phase='REPORT_COMPLETE',utc=utc(),pid=os.getpid()))
    print('REPORT_COMPLETE',json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',required=True)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--smoke',action='store_true')
    stages=dict(threshold=threshold_stage,continuation=continuation_stage,fragment_data=fragment_data_stage,
        fragment_graph=fragment_graph_stage,contour_data=contour_data_stage,contour_graph=contour_graph_stage,report=report_stage)
    p.add_argument('--stage',required=True,choices=list(stages));args=p.parse_args()
    try:
        c,out,index,contract=prepare(args);announce(out,args.stage.upper())
        stages[args.stage](c,out,index,contract,args.smoke)
    except Exception as exc:record_failure(args.output.resolve(),exc);raise


def contour_path(out,phase,sid):return out/'contour_data'/phase/sid.replace(':','__')


def contour_data_stage(c,out,index,contract,smoke):
    directory=out/'contour_data';directory.mkdir(exist_ok=False)
    model=source_model(c,contract).requires_grad_(False);ds,by=dataset_for(c,contract)
    transform=_transform(contract['resolved_preprocessing'],'letterbox',train=False)
    records,_=source(c,index);config=c['contour'];count=0;supervised=0;ignored=0;multiple=0
    torch.cuda.reset_peak_memory_stats()
    for phase,entries in index.items():
        for entry in entries:
            sid=entry['sample_id'];record=copy.deepcopy(records[phase][sid]);rgb=load_rgb(entry)
            with torch.no_grad():evidence=model.forward_evidence(_tensor(rgb,transform).cuda())
            prob=evidence['words'][0].sigmoid().cpu().numpy()
            if file_sha256(Path(record['maps_path']))!=record['maps_sha256']:raise ValueError('Original contour map changed')
            with np.load(record['maps_path']) as z:
                if not np.array_equal(prob,z['probabilities']):raise ValueError('Contour backbone replay failed')
            canvas=F.interpolate(torch.cat((evidence['features'],evidence['words'].sigmoid()),1),
                size=(config['feature_size'],config['feature_size']),mode='bilinear',align_corners=False)[0].cpu().numpy()
            pool=record['arms']['mk']['pool'];boxes=np.array([p['bbox_native'] for p in pool],np.float32).reshape(len(pool),4)
            initial=rectangle_points(boxes,config['points']);arrays=dict(feature_map=canvas,probabilities=prob,boxes=boxes,initial=initial)
            if phase=='fit':
                batch=ds[by[sid]];owners=batch['owners_native'].numpy();validity=batch['validity'][0].numpy().astype(bool)
                target=initial.copy();valid=np.zeros(len(pool),bool)
                for j,box in enumerate(boxes.astype(int)):
                    x0,y0,x1,y1=box;patch=owners[y0:y1,x0:x1];ok=validity[y0:y1,x0:x1]
                    ids,sizes=np.unique(patch[ok & (patch>0)],return_counts=True)
                    if not len(ids) or ok.mean()<config['minimum_valid_fraction']:ignored+=1;continue
                    identity=int(ids[sizes.argmax()]);yy,xx=np.where(owners==identity)
                    target_box=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]
                    if iou(box.tolist(),target_box)<config['minimum_training_iou']:ignored+=1;continue
                    points,components=resample_boundary(owners==identity,config['points'])
                    multiple+=int(components>1)
                    if points is None:ignored+=1;continue
                    target[j]=points;valid[j]=True;supervised+=1
                arrays.update(target=target,valid=valid)
            target_path=contour_path(out,phase,sid);target_path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target_path.with_suffix('.npz'),**arrays)
            write(target_path.with_suffix('.json'),dict(record=record,arrays_sha256=file_sha256(target_path.with_suffix('.npz'))))
            count+=1
            if count%50==0:announce(out,'CONTOUR_DATA',images=count,total=sum(map(len,index.values())))
    write(directory/'memory.json',peaks('Frozen U-Net; cached128-square CNN/probability field for contour CPU sampling',1,[1,3,512,512]))
    write(directory/'complete.json',dict(status='SUCCESS',images=count,supervised_fit_proposals=supervised,
        unsupervised_fit_proposals=ignored,training_targets_with_multiple_components=multiple))
    del model,evidence;release()


def load_contour(out,phase,sid):
    path=contour_path(out,phase,sid);meta=read(path.with_suffix('.json'))
    if file_sha256(path.with_suffix('.npz'))!=meta['arrays_sha256']:raise ValueError('Contour data changed')
    with np.load(path.with_suffix('.npz')) as z:arrays={k:z[k].copy() for k in z.files}
    return meta,arrays


def contour_records(model,c,out,entries,phase,contract,trace):
    values={};model.eval();trace.mkdir(parents=True,exist_ok=True)
    pp=contract['resolved_config']['postprocessing']
    for entry in entries:
        sid=entry['sample_id'];meta,z=load_contour(out,phase,sid);record=meta['record'];initial=torch.from_numpy(z['initial']).cuda();boxes=torch.from_numpy(z['boxes']).cuda()
        with torch.no_grad():points=model(torch.from_numpy(z['feature_map'])[None],initial,boxes,z['probabilities'].shape[1]).cpu().numpy()
        adjusted=contour_boxes(points,z['initial'],z['boxes'],z['probabilities'].shape[1]);rgb=load_rgb(entry)
        for proposal,box in zip(record['arms']['mk']['pool'],adjusted):
            proposal['bbox_native']=box
            original=remap_letterbox_native_box(box,rgb.shape[:2][::-1],native_size=z['probabilities'].shape[1])
            proposal['bbox']=original;proposal['padded_bbox_original']=original.copy()
            x0,y0,x1,y1=original
            # The trace is saved before budget_and_filter refreshes hashes.
            # Bind it to the deformed pixels rather than the inherited source crop.
            proposal['crop_array_sha256']=crop_hash(np.ascontiguousarray(rgb[y0:y1,x0:x1]))
        record['features']['mk']={p['id']:features(p,rgb,z['probabilities'][0],text_threshold=pp['text_threshold']) for p in record['arms']['mk']['pool']}
        values[sid]=record
        write(trace/(sid.replace(':','__')+'.json'),dict(record=record,initial_points=z['initial'].tolist(),predicted_points=points.tolist()))
    return values


def contour_graph_stage(c,out,index,contract,smoke):
    directory=out/'contour_graph';directory.mkdir(exist_ok=False);config=c['contour']
    records,readings=source(c,index);ocr=CropOCR(c,directory/'ocr.jsonl');ocr.rows.update(readings)
    initial_pred=baseline_filter(c,records['calibration'],'calibration',ocr)
    initial_f1=report(records['calibration'],initial_pred,ocr.rows)['aggregate']['f1']
    train=[]
    for entry in index['fit']:
        _,z=load_contour(out,'fit',entry['sample_id'])
        if z['valid'].any():train.append(z)
    if not train:raise ValueError('No valid contour training targets')
    selections={};capacities={}
    for arm in ('self','gnn'):
        seed_everything(c['seed'],deterministic_algorithms=True)
        model=ContourPointRefiner(train[0]['feature_map'].shape[0],config['hidden'],config['layers'],
            message_passing=arm=='gnn',iterations=config['iterations'],maximum_step=config['maximum_step']).cuda()
        capacities[arm]=sum(p.numel() for p in model.parameters())
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        generator=torch.Generator().manual_seed(c['seed']);branch=directory/arm;branch.mkdir();history=[]
        best=dict(epoch=0,f1=initial_f1);torch.cuda.reset_peak_memory_stats();epochs=1 if smoke else config['epochs']
        for epoch in range(1,epochs+1):
            model.train();losses=[]
            for j in torch.randperm(len(train),generator=generator).tolist():
                z=train[j];optimizer.zero_grad(set_to_none=True)
                points=torch.from_numpy(z['initial']).cuda();boxes=torch.from_numpy(z['boxes']).cuda()
                prediction=model(torch.from_numpy(z['feature_map'])[None],points,boxes,z['probabilities'].shape[1])
                loss=cyclic_contour_loss(prediction,torch.from_numpy(z['target']).cuda(),torch.from_numpy(z['valid']).cuda(),boxes,beta=config['loss_beta'])
                if not loss.isfinite():raise FloatingPointError('Nonfinite contour loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();losses.append(float(loss.detach()))
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),updates=len(losses)));write(branch/'history.json',history)
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch),branch/'last.pt');announce(out,'CONTOUR_TRAIN',arm=arm,**history[-1])
            if epoch%config['calibration_every']==0 or epoch==epochs:
                exported=contour_records(model,c,out,index['calibration'],'calibration',contract,branch/f'epoch{epoch}_calibration')
                predictions=budget_and_filter(c,exported,ocr);metric=report(exported,predictions,ocr.rows)
                write(branch/f'epoch{epoch}_calibration/results.json',metric)
                if metric['aggregate']['f1']>best['f1']:
                    best=dict(epoch=epoch,f1=metric['aggregate']['f1']);torch.save(dict(state_dict=model.state_dict(),selection=best),branch/'best.pt')
                announce(out,'CONTOUR_CALIBRATION',arm=arm,epoch=epoch,f1=metric['aggregate']['f1'],best=best);ocr.release()
        selections[arm]=best;write(branch/'selection_frozen.json',best)
        write(branch/'memory.json',peaks('Contour refiner + AdamW; CPU feature maps and differentiable CPU grid sampling; PARSeq at calibration',1,['variable_proposals',config['points'],66]))
        for name,checkpoint in ((arm,branch/'best.pt' if best['epoch'] else None),(arm+'_final',branch/'last.pt')):
            if checkpoint is not None:model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'],strict=True)
            for phase in ('evaluation','test'):
                if checkpoint is None:exported=records[phase];predictions=baseline_filter(c,exported,phase,ocr)
                else:
                    exported=contour_records(model,c,out,index[phase],phase,contract,branch/name/phase)
                    predictions=budget_and_filter(c,exported,ocr)
                save_endpoint(directory,name,phase,exported,predictions,ocr)
        del model,optimizer;ocr.release();release()
    if capacities['self']!=capacities['gnn']:raise ValueError('Contour controls have unequal capacity')
    write(directory/'complete.json',dict(status='SUCCESS',selections=selections,parameters=capacities))


if __name__=='__main__':main()
