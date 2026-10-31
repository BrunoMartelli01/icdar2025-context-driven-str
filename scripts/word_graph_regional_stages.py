"""FIT-verified complete-word region distillation: GT, frozen and EMA controls."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from localize.word_graph_recovery import regional_distillation
from localize.graph_occlusion_research import occlude_instances
from localize.teacher_student_tests import ema_update
from localize.occlusion_four_tests import iou
from scripts.run_word_graph_recovery import initial,announce,checkpoint,save_result,memory,training_batch
from scripts.run_graph_occlusion_research import dataset_for,budget_and_filter,recognized_segmentation
from scripts.run_teacher_student_tests import source_model,supervised_loss,export_model,report,release,frozen_selector
from scripts.run_occlusion_four_tests import load_rgb,labels_for,preselect,score_record
from scripts.run_mask_crop_budget import write
from scripts.evaluate_cached_totaltext_padding import crop_hash
from segmentation.training import seed_everything


def verify_fit_regions(records,probabilities,batches,words,readings,config):
    """Annotations VERIFY already proposed/read FIT crops, never held-out inputs.

    Reject incomplete/mixed-word teacher support as well as wrong transcriptions.
    Unknown regions create no distillation negatives. GT supervision is separate.
    """
    accepted={};audit={}
    for sid,r in records.items():
        owner=batches[sid]['owners_native'].numpy();probability=probabilities[sid][0];rows=[];seen=set()
        counts=dict(candidates=0,wrong_text_or_geometry=0,low_confidence=0,impure_or_incomplete=0,duplicate_owner=0)
        for p in r['verified_admitted']:
            counts['candidates']+=1;reading=readings[p['crop_array_sha256']]
            confidence=reading.get('token_probabilities',[])
            if not confidence or float(np.mean(confidence))<config['minimum_confidence']:
                counts['low_confidence']+=1;continue
            matches=[w for w in words[sid] if iou(p['bbox'],w['bbox'])>=config['minimum_iou'] and reading['raw_text'].casefold()==w['text'].casefold()]
            if not matches:
                counts['wrong_text_or_geometry']+=1;continue
            x0,y0,x1,y1=map(int,p['bbox_native']);patch=owner[y0:y1,x0:x1]
            foreground=probability[y0:y1,x0:x1]>=config['text_threshold']
            ids,amount=np.unique(patch[foreground & (patch>0)],return_counts=True)
            identity=int(ids[amount.argmax()]) if len(ids) else 0
            correct=int((foreground & (patch==identity)).sum()) if identity else 0
            purity=correct/max(1,int(foreground.sum()));coverage=correct/max(1,int((owner==identity).sum()))
            if not identity or purity<config['minimum_purity'] or coverage<config['minimum_coverage']:
                counts['impure_or_incomplete']+=1;continue
            if identity in seen:
                counts['duplicate_owner']+=1;continue
            seen.add(identity);rows.append(dict(bbox_native=p['bbox_native'],pseudo_text=reading['raw_text'],
                owner_id=identity,purity=purity,coverage=coverage,crop_array_sha256=p['crop_array_sha256']))
        accepted[sid]=rows;audit[sid]={**counts,'accepted':len(rows)}
    return accepted,audit


def refresh_regions(teacher,c,contract,index,batches,fit_words,ocr,directory):
    teacher.eval();records=export_model(teacher,c,contract,index['fit'],directory/'maps')
    pre=frozen_selector(c,'pre');probabilities={}
    for sid,r in records.items():
        rgb=load_rgb(r)
        for p in r['arms']['mk']['pool']:
            x0,y0,x1,y1=p['padded_bbox_original'];p['crop_array_sha256']=crop_hash(np.ascontiguousarray(rgb[y0:y1,x0:x1]))
        ids=preselect(r,'mk',score_record(r,'mk',ocr,pre,False),c['budget'])
        pool=[p for p in r['arms']['mk']['pool'] if p['id'] in ids]
        if len({p['crop_array_sha256'] for p in pool})>c['budget']:raise ValueError('Teacher OCR budget overflow')
        ocr.image(rgb,pool);r['verified_admitted']=pool
        with np.load(directory/'maps'/(sid.replace(':','__')+'.npz')) as z:probabilities[sid]=z['probabilities'].copy()
    regions,audit=verify_fit_regions(records,probabilities,batches,fit_words,ocr.rows,c['regional_v1'])
    write(directory/'regions.json',regions);write(directory/'verification.json',dict(scope='FIT only',per_image=audit,
        accepted=sum(len(v) for v in regions.values()),unique_crop_budget_per_image=c['budget']))
    ocr.release();return regions


def regional_stage(c,out,index,contract,smoke):
    directory=out/'regional';directory.mkdir(exist_ok=False);g=c['regional_v1']
    records,ocr,baseline=initial(c,index,directory);fit_words,_=labels_for(records['fit']);ds,by=dataset_for(c,contract)
    batches={entry['sample_id']:ds[by[entry['sample_id']]] for entry in index['fit']};sids=list(batches)
    initial_f1=report(records['calibration'],baseline['calibration'],ocr.rows)['aggregate']['f1'];audit={}
    for kind in ('gt','fixed','ema'):
        seed_everything(c['seed'],deterministic_algorithms=True);branch=directory/kind;branch.mkdir();torch.cuda.reset_peak_memory_stats()
        student=source_model(c,contract);teacher=copy.deepcopy(student).requires_grad_(False).eval() if kind!='gt' else None
        original={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
        opt=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=g['learning_rate'],weight_decay=g['weight_decay'])
        rng=torch.Generator().manual_seed(c['seed']);epochs=1 if smoke else g['epochs'];history=[];best=dict(epoch=0,f1=initial_f1)
        regions={sid:[] for sid in sids}
        for epoch in range(1,epochs+1):
            if teacher is not None and (epoch==1 or (epoch-1)%g['refresh_every']==0):
                regions=refresh_regions(teacher,c,contract,index,batches,fit_words,ocr,branch/f'epoch{epoch}_teacher')
                announce(out,'REGIONAL_TEACHER',arm=kind,epoch=epoch,accepted_regions=sum(map(len,regions.values())))
            student.train();losses=[];extras=[];order=torch.randperm(len(sids),generator=rng).tolist()
            for start in range(0,len(order),g['batch_size']):
                ids=[sids[j] for j in order[start:start+g['batch_size']]]
                batch=training_batch([batches[sid] for sid in ids],device='cuda:0')
                # No horizontal flip: teacher OCR verifies ordinary clean text.
                contrast=1+float(torch.rand((),generator=rng)*2-1)*g['contrast'];brightness=float(torch.rand((),generator=rng)*2-1)*g['brightness']
                strong,_=occlude_instances(batch['image']*contrast+brightness,batch['owners_native'],rng,
                    probability=g['occlusion_probability'],width_fraction=g['occlusion_width_fraction'],maximum_words=g['maximum_occluded_words'])
                opt.zero_grad(set_to_none=True);logits=student(strong);loss=supervised_loss(logits,batch,contract['resolved_config']['loss'])
                extra=logits.sum()*0
                if teacher is not None:
                    with torch.no_grad():target=teacher(batch['image']).sigmoid()
                    extra=regional_distillation(logits,target,[regions[sid] for sid in ids],temperature=g['extent_temperature'],extent_weight=g['extent_weight'])
                    loss=loss+g['regional_weight']*extra
                if not loss.isfinite():raise FloatingPointError('Nonfinite regional training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),g['gradient_clip'],error_if_nonfinite=True);opt.step()
                if kind=='ema':ema_update(teacher,student,g['ema_decay'])
                losses.append(float(loss.detach()));extras.append(float(extra.detach()))
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),regional_loss=float(np.mean(extras)),updates=len(losses),accepted_regions=sum(map(len,regions.values()))))
            write(branch/'history.json',history);checkpoint(branch/'last.pt',state_dict=student.state_dict(),optimizer=opt.state_dict(),
                generator_state=rng.get_state(),teacher_state_dict=teacher.state_dict() if teacher is not None else {},epoch=epoch)
            if epoch%g['calibration_every']==0 or epoch==epochs:
                _,_,metric=recognized_segmentation(student.eval(),c,contract,index['calibration'],branch/f'epoch{epoch}_calibration',ocr)
                write(branch/f'epoch{epoch}_calibration/results.json',metric)
                if metric['aggregate']['f1']>best['f1']:
                    best=dict(epoch=epoch,f1=metric['aggregate']['f1']);checkpoint(branch/'best.pt',state_dict=student.state_dict(),selection=best)
                ocr.release()
            announce(out,'REGIONAL_TRAIN',arm=kind,**history[-1],best=best)
        encoder_changed=[k for k,v in student.state_dict().items() if k.startswith('encoder.') and not torch.equal(v.cpu(),original[k])]
        updated=[k for k,v in student.state_dict().items() if not torch.equal(v.cpu(),original[k])]
        teacher_changed=[] if teacher is None else [k for k,v in teacher.state_dict().items() if not torch.equal(v.cpu(),original[k])]
        if encoder_changed or not updated or (kind=='fixed' and teacher_changed):raise ValueError('Regional frozen-state/update contract failed')
        if kind=='ema' and not teacher_changed:raise ValueError('EMA teacher did not update')
        if kind!='gt' and not any(v['accepted_regions'] for v in history):raise ValueError('No verified teacher regions: cannot qualify regional distillation')
        audit[kind]=dict(encoder_changed=encoder_changed,student_changed_tensors=len(updated),teacher_changed_tensors=len(teacher_changed))
        write(branch/'selection_frozen.json',best)
        for label,path in ((kind+'_final',branch/'last.pt'),(kind,branch/'best.pt' if best['epoch'] else Path(c['paths']['source_checkpoint']))):
            saved=torch.load(path,map_location='cpu',weights_only=False)
            student.load_state_dict(saved['state_dict'] if 'state_dict' in saved else saved['model_state_dict'],strict=True)
            for phase in ('evaluation','test'):
                exported,pp,_=recognized_segmentation(student.eval(),c,contract,index[phase],branch/label/phase,ocr)
                save_result(directory,label,phase,exported,pp,ocr)
        memory(branch,'U-Net student + optional frozen/EMA U-Net teacher + AdamW; frozen PARSeq during teacher verification/calibration',g['batch_size'],[g['batch_size'],3,512,512])
        del student,teacher,opt,original;ocr.release();release()
    for phase in ('evaluation','test'):save_result(directory,'baseline',phase,records[phase],baseline[phase],ocr)
    write(directory/'complete.json',dict(status='SUCCESS',state_audit=audit,
        objective='Supervised GT in all arms; FIT-verified regional Dice/soft-extrema in fixed/EMA arms. No pseudo-background on unaccepted regions.'))
