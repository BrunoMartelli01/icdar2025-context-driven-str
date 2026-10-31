"""True U-Net-free proposal/deformation training and matched fresh controls."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from localize.word_graph_recovery import geometry_targets, dense_geometry_loss, training_owner_targets, complete_word_loss
from localize.graph_occlusion_followups import (ContourPointRefiner, rectangle_points,
    resample_boundary, cyclic_contour_loss, contour_boxes)
from localize.occlusion_four_tests import decode_instance_labels, instance_pool, features
from models.word_graph_decoder import GeometricProposalDecoder, UNetGeometryControl
from models.adaspot_gap import AdaSpotGapAGUNet
from scripts.run_word_graph_recovery import (initial, announce, checkpoint, save_result, memory, training_batch)
from scripts.run_graph_occlusion_research import budget_and_filter, dataset_for
from scripts.run_teacher_student_tests import supervised_loss, report, release
from scripts.run_occlusion_four_tests import load_rgb
from scripts.run_mask_crop_budget import read,write
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _tensor
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from scripts.evaluate_cached_totaltext_padding import crop_hash
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask, remap_letterbox_native_box
from segmentation.training import seed_everything


class GeometrySystem(nn.Module):
    """A proposal CNN and optional contour refiner; inference NEVER accepts GT."""
    def __init__(self, proposal, config, kind):
        super().__init__();self.proposal=proposal;self.kind=kind
        self.refiner=ContourPointRefiner(69,config['hidden'],config['layers'],message_passing=kind=='graph',
            iterations=config['iterations'],maximum_step=config['maximum_step']) if kind in ('self','graph') else None


def source_encoder_state(c,contract):
    checkpoint=torch.load(c['paths']['source_checkpoint'],map_location='cpu',weights_only=False)
    if checkpoint['run_contract']!=contract:raise ValueError('Qualified encoder contract changed')
    state=checkpoint['model_state_dict']
    encoder={k.removeprefix('encoder.'):v.clone() for k,v in state.items() if k.startswith('encoder.')}
    if not encoder:raise ValueError('Missing complete source encoder')
    return encoder


def build_system(c,contract,encoder,kind):
    g=c['geometry_v1'];seed_everything(c['seed'],deterministic_algorithms=True)
    if kind=='unet':
        fresh=AdaSpotGapAGUNet('control',skip_floor=contract['resolved_config']['skip_floor'],initial_gap_bias=g['unused_gap_bias'])
        proposal=UNetGeometryControl(fresh,encoder)
    else:proposal=GeometricProposalDecoder(encoder,channels=g['lateral_channels'],native_size=g['native_size'])
    return GeometrySystem(proposal,g,kind).cuda()


def inferred_boxes(evidence,canvas,pp):
    probability=evidence['words'][0].sigmoid().detach().cpu().numpy()
    labels=decode_instance_labels(probability,None,canvas,pp,None,'cut')
    boxes=[]
    for identity in sorted(np.unique(labels[labels>0])):
        yy,xx=np.where(labels==identity)
        boxes.append([int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1])
    return probability,labels,np.asarray(boxes,np.float32).reshape(-1,4)


def contour_field(evidence):
    geometry=evidence['geometry']
    direction=F.normalize(geometry[:,1:],dim=1,eps=1e-6)
    return torch.cat((evidence['features'],evidence['words'].sigmoid(),geometry[:,:1].sigmoid(),direction),1).cpu()


def graph_fit_loss(system,evidence,batch,c,contract):
    """Only predictions propose windows. Owners label those windows during FIT."""
    g=c['geometry_v1'];terms=[];valid_count=0
    for j in range(len(batch['image'])):
        one={k:v[j:j+1] for k,v in evidence.items() if isinstance(v,torch.Tensor)}
        probability,labels,boxes=inferred_boxes(one,batch['canvas'][j].cpu().numpy().astype(bool),contract['resolved_config']['postprocessing'])
        if not len(boxes):continue
        owners=batch['owners_native'][j].detach().cpu().numpy();validity=batch['validity'][j,0].detach().cpu().numpy()
        target_box,ids,valid=training_owner_targets(boxes,owners,validity,
            minimum_iou=g['minimum_training_iou'],minimum_valid_fraction=g['minimum_valid_fraction'])
        if not valid.any():continue
        initial=rectangle_points(boxes,g['points']);target=initial.copy()
        for k in np.flatnonzero(valid):
            sampled,_=resample_boundary(owners==ids[k],g['points'])
            if sampled is None:valid[k]=False
            else:target[k]=sampled
        if not valid.any():continue
        b=torch.from_numpy(boxes).to(evidence['words'].device);a=torch.from_numpy(initial).to(b)
        predicted=system.refiner(contour_field(one),a,b,g['native_size'])
        valid_tensor=torch.from_numpy(valid).to(b.device)
        loss=cyclic_contour_loss(predicted,torch.from_numpy(target).to(b),valid_tensor,b,beta=g['loss_beta'])
        loss=loss+complete_word_loss(predicted,b,torch.from_numpy(target_box).to(b),torch.from_numpy(ids).to(b.device),
            batch['owners_native'][j],valid_tensor,inside_weight=g['inside_weight'],extent_weight=g['extent_weight'],contamination_weight=g['contamination_weight'])
        terms.append(loss);valid_count+=int(valid.sum())
    return (torch.stack(terms).mean() if terms else evidence['words'].sum()*0),valid_count


def geometry_records(system,c,contract,entries,directory):
    directory.mkdir(parents=True,exist_ok=True);system.eval();g=c['geometry_v1']
    transform=_transform(contract['resolved_preprocessing'],'letterbox',train=False)
    pp=contract['resolved_config']['postprocessing'];hyp=read(Path(c['source_run'])/'resolved_config.json')['hypotheses'];result={}
    for entry in entries:
        sid=entry['sample_id'];rgb=load_rgb(entry)
        canvas=np.asarray(letterbox_canvas_mask((rgb.shape[1],rgb.shape[0]),native_size=g['native_size']),bool)
        with torch.no_grad():
            evidence=system.proposal.forward_evidence(_tensor(rgb,transform).cuda())
            probability,labels,boxes=inferred_boxes(evidence,canvas,pp)
            pool=instance_pool(probability[0],labels,rgb,hyp)
            if system.refiner is not None:
                boxes=np.asarray([p['bbox_native'] for p in pool['pool']],np.float32).reshape(-1,4)
                a=rectangle_points(boxes,g['points'])
                predicted=system.refiner(contour_field(evidence),torch.from_numpy(a).cuda(),torch.from_numpy(boxes).cuda(),g['native_size']).cpu().numpy()
                adjusted=contour_boxes(predicted,a,boxes,g['native_size'])
                for p,box in zip(pool['pool'],adjusted):
                    p['bbox_native']=box;p['bbox']=remap_letterbox_native_box(box,rgb.shape[:2][::-1],native_size=g['native_size'])
                    p['padded_bbox_original']=p['bbox'].copy()
            for p in pool['pool']:
                x0,y0,x1,y1=p['padded_bbox_original'];p['crop_array_sha256']=crop_hash(np.ascontiguousarray(rgb[y0:y1,x0:x1]))
            r={**entry,'image_hw':list(rgb.shape[:2]),'arms':{'mk':pool},'features':{'mk':{
                p['id']:features(p,rgb,probability[0],text_threshold=pp['text_threshold']) for p in pool['pool']}}}
            result[sid]=r
            np.savez_compressed(directory/(sid.replace(':','__')+'.npz'),probabilities=probability,geometry=evidence['geometry'][0].cpu().numpy(),labels=labels)
    write(directory/'records.json',result);return result


def geometry_stage(c,out,index,contract,smoke):
    directory=out/'geometry';directory.mkdir(exist_ok=False);g=c['geometry_v1']
    source,ocr,baseline=initial(c,index,directory);ds,by=dataset_for(c,contract);encoder=source_encoder_state(c,contract)
    batches=[]
    for entry in index['fit']:
        batch=dict(ds[by[entry['sample_id']]])
        batch['geometry']=torch.from_numpy(geometry_targets(batch['owners_native'].numpy()))
        rgb=load_rgb(entry);batch['canvas']=torch.from_numpy(np.asarray(letterbox_canvas_mask((rgb.shape[1],rgb.shape[0]),native_size=g['native_size']),bool))
        batches.append(batch)
    parameters={};initial_proposals=None
    for kind in ('unet','proposal','self','graph'):
        branch=directory/kind;branch.mkdir();system=build_system(c,contract,encoder,kind);torch.cuda.reset_peak_memory_stats()
        frozen={k:v.detach().cpu().clone() for k,v in system.proposal.state_dict().items() if ('encoder.' in k)}
        parameters[kind]=dict(total=sum(p.numel() for p in system.parameters()),trainable=sum(p.numel() for p in system.parameters() if p.requires_grad),requires_unet=kind=='unet')
        # The proposal part of all three new systems must start identically.
        if kind!='unet':
            now={k:v.detach().cpu().clone() for k,v in system.proposal.state_dict().items()}
            if initial_proposals is None:initial_proposals=now
            elif any(not torch.equal(v,initial_proposals[k]) for k,v in now.items()):raise ValueError('Unmatched geometric proposal initialization')
        optimizer=torch.optim.AdamW([p for p in system.parameters() if p.requires_grad],lr=g['learning_rate'],weight_decay=g['weight_decay'])
        rng=torch.Generator().manual_seed(c['seed']);best=dict(epoch=-1,f1=-1.);history=[];epochs=1 if smoke else g['epochs']
        for epoch in range(1,epochs+1):
            system.train();losses=[];contours=0;order=torch.randperm(len(batches),generator=rng).tolist()
            for start in range(0,len(order),g['batch_size']):
                selected=[batches[j] for j in order[start:start+g['batch_size']]]
                batch=training_batch(selected,device='cuda:0',extra=('geometry','canvas'))
                optimizer.zero_grad(set_to_none=True);evidence=system.proposal.forward_evidence(batch['image'])
                loss=supervised_loss(evidence['words'],batch,contract['resolved_config']['loss'])
                loss=loss+g['geometry_weight']*dense_geometry_loss(evidence['geometry'],batch['geometry'],batch['mask'],batch['validity'],magnitude_weight=g['magnitude_weight'])
                if system.refiner is not None:
                    extra,count=graph_fit_loss(system,evidence,batch,c,contract);loss=loss+g['contour_weight']*extra;contours+=count
                if not loss.isfinite():raise FloatingPointError('Nonfinite U-Net-free training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(system.parameters(),g['gradient_clip'],error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),updates=len(losses),supervised_predicted_contours=contours));write(branch/'history.json',history)
            checkpoint(branch/'last.pt',state_dict=system.state_dict(),optimizer=optimizer.state_dict(),generator_state=rng.get_state(),epoch=epoch,requires_unet=kind=='unet')
            if epoch%g['calibration_every']==0 or epoch==epochs:
                exported=geometry_records(system,c,contract,index['calibration'],branch/f'epoch{epoch}_calibration')
                pp=budget_and_filter(c,exported,ocr);metric=report(exported,pp,ocr.rows);write(branch/f'epoch{epoch}_calibration/results.json',metric)
                if metric['aggregate']['f1']>best['f1']:
                    best=dict(epoch=epoch,f1=metric['aggregate']['f1'],requires_unet=kind=='unet')
                    checkpoint(branch/'best.pt',state_dict=system.state_dict(),selection=best)
                ocr.release()
            announce(out,'GEOMETRY_TRAIN',arm=kind,**history[-1],best=best)
        changed=[k for k,v in system.proposal.state_dict().items() if k in frozen and not torch.equal(v.cpu(),frozen[k])]
        if changed:raise ValueError('Frozen shared encoder changed: '+str(changed))
        if system.refiner is not None and not smoke and not any(row['supervised_predicted_contours'] for row in history):
            raise ValueError('No valid predicted FIT contour reached the graph. Dense proposal bootstrap failed; this is NOT a trained graph result.')
        if system.refiner is not None:
            saved=torch.load(branch/'last.pt',map_location='cpu',weights_only=True)['state_dict']
            changed_graph=sum(not torch.equal(v.cpu(),saved[k].cpu()) for k,v in system.state_dict().items())
            if changed_graph:raise ValueError('Checkpoint round trip changed tensors')
        write(branch/'selection_frozen.json',best);write(branch/'encoder_audit.json',dict(status='PASS',changed_encoder_tensors=changed,requires_unet=kind=='unet',
            supervised_predicted_contours=sum(row['supervised_predicted_contours'] for row in history),
            graph_supervision=('PASS' if any(row['supervised_predicted_contours'] for row in history) else 'NONE_SMOKE') if system.refiner is not None else 'NOT_APPLICABLE'))
        for label,path in ((kind+'_final',branch/'last.pt'),(kind,branch/'best.pt')):
            system.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'],strict=True)
            for phase in ('evaluation','test'):
                exported=geometry_records(system,c,contract,index[phase],branch/label/phase)
                save_result(directory,label,phase,exported,budget_and_filter(c,exported,ocr),ocr)
        memory(branch,'Frozen ResNet50 + trainable decoder/geometry/optional contour + AdamW; CPU bilinear backward; frozen PARSeq at calibration',g['batch_size'],[g['batch_size'],3,512,512])
        del system,optimizer;ocr.release();release()
    if parameters['self']!=parameters['graph']:raise ValueError('Unmatched self/graph decoder capacity')
    for phase in ('evaluation','test'):save_result(directory,'baseline',phase,source[phase],baseline[phase],ocr)
    write(directory/'complete.json',dict(status='SUCCESS',parameters=parameters,
        comparison='Fresh decoders share the frozen qualified encoder, dense geometry objectives, FIT data and update budget. Original warm baseline is an operational reference. Refiner losses add supervision to self/graph arms.'))
