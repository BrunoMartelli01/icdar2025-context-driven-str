"""Calibration-free real-RGB CUDA gradient probe, NOT a spotting result.

Fixed archived FIT proposals are an explicit interface fixture. They are never
substituted for the U-Net-free system's predicted proposals during experiments.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.run_word_graph_recovery import prepare,contour_payload,memory
from scripts.word_graph_geometry_stages import build_system,source_encoder_state,contour_field
from scripts.run_graph_occlusion_research import dataset_for,record_failure
from localize.graph_occlusion_followups import cyclic_contour_loss
from localize.word_graph_recovery import training_owner_targets,complete_word_loss
from scripts.run_mask_crop_budget import write,utc


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',required=True)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args();args.smoke=True
    try:
        c,out,index,contract=prepare(args);g=c['geometry_v1'];ds,by=dataset_for(c,contract)
        entry=index['fit'][0];sid=entry['sample_id'];batch=ds[by[sid]];_,z=contour_payload(c,'fit',sid)
        if not z['valid'].any():raise ValueError('Predeclared first-FIT fixture has no qualified contour targets')
        target_box,ids,valid=training_owner_targets(z['boxes'],batch['owners_native'].numpy(),batch['validity'][0].numpy(),
            minimum_iou=g['minimum_training_iou'],minimum_valid_fraction=g['minimum_valid_fraction'])
        system=build_system(c,contract,source_encoder_state(c,contract),'graph').train()
        if system.proposal.requires_unet:raise ValueError('Unexpected U-Net decoder in geometry probe')
        optimizer=torch.optim.AdamW([p for p in system.parameters() if p.requires_grad],lr=g['learning_rate'])
        before={k:v.detach().cpu().clone() for k,v in system.refiner.state_dict().items()};steps=[];torch.cuda.reset_peak_memory_stats()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True);evidence=system.proposal.forward_evidence(batch['image'][None].cuda())
            boxes=torch.from_numpy(z['boxes']).cuda();initial=torch.from_numpy(z['initial']).cuda()
            points=system.refiner(contour_field(evidence),initial,boxes,g['native_size'])
            loss=cyclic_contour_loss(points,torch.from_numpy(z['target']).cuda(),torch.from_numpy(z['valid']).cuda(),boxes,beta=g['loss_beta'])
            loss=loss+complete_word_loss(points,boxes,torch.from_numpy(target_box).cuda(),torch.from_numpy(ids).cuda(),batch['owners_native'].cuda(),
                torch.from_numpy(valid).cuda(),inside_weight=g['inside_weight'],extent_weight=g['extent_weight'],contamination_weight=g['contamination_weight'])
            loss.backward()
            gradients={name:float(parameter.grad.abs().sum()) if parameter.grad is not None else 0. for name,parameter in system.named_parameters()}
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in system.parameters()):raise FloatingPointError('Nonfinite real-RGB graph gradients')
            steps.append(dict(step=step+1,loss=float(loss.detach()),graph_offset_gradient=gradients['refiner.offset.weight'],
                proposal_gradient=sum(v for k,v in gradients.items() if k.startswith('proposal.') and '.encoder.' not in k)))
            torch.nn.utils.clip_grad_norm_(system.parameters(),g['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        if steps[-1]['graph_offset_gradient']<=0 or steps[-1]['proposal_gradient']<=0:raise ValueError('CNN/graph gradient boundary failed')
        if any(p.grad is not None for p in system.proposal.encoder.parameters()):raise ValueError('Frozen encoder acquired gradients')
        changed=sum(not torch.equal(value.cpu(),before[k]) for k,value in system.refiner.state_dict().items())
        if not changed:raise ValueError('Refiner tensors did not update')
        memory(out,'U-Net-free ResNet50 geometry CNN + GCN + AdamW, real FIT RGB, archived proposal interface fixture',1,[1,3,512,512])
        write(out/'gradient_probe.json',dict(status='PASS',classification='ASSUMPTION',scope='Real-RGB component gradient test, not learned proposal quality or word-F1 efficacy',
            sample_id=sid,fixture='Archived FIT native proposal boxes and owner-labelled contour targets; no substitution in the experiment',
            steps=steps,changed_refiner_tensors=changed,requires_unet_inference=False,frozen_encoder_gradients='NONE',uses_calibration_or_test=False))
        write(out/'status.json',dict(status='SUCCESS',phase='GRADIENT_PROBE_COMPLETE',utc=utc()))
        print('GRADIENT_PROBE_PASS',steps,flush=True)
    except Exception as exc:record_failure(args.output.resolve(),exc);raise


if __name__=='__main__':main()
