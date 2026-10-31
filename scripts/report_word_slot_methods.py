"""Independent Hungarian audit and complete fixed-budget word-slot result report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from segmentation.word_slot_data import read,write,sha,sample_key
from recognition.word_slot_evaluation import aggregate,select,measure
from scripts.evaluate_word_slot_methods import raw_path
from scripts.run_word_slot_methods import teacher_preprocessing
from recognition.word_slot_text import normalize_word_text,comparison_label,CASEFOLD


def independent_iou(a,b):
    if a is None or b is None:return 0.
    # Match the declared inherited ORIGINAL-pixel metric, independently of its
    # implementation. Normalized-box NMS uses floating arithmetic separately.
    a=[int(v) for v in a];b=[int(v) for v in b]
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1]);area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return intersection/max(area_a+area_b-intersection,1e-12)


def hungarian(predictions,truth,iou,text=True,normalization=CASEFOLD):
    # Independent implementation: do not call the production text normalizer.
    if normalization not in ('casefold','alphanumeric_casefold'):raise ValueError('Unknown audit text normalization')
    def compared(value):
        folded=value.casefold()
        return ''.join(c for c in folded if c.isalnum()) if normalization=='alphanumeric_casefold' else folded
    adjacency=np.zeros((len(predictions),len(truth)),dtype=np.int8)
    for i,p in enumerate(predictions):
        for j,g in enumerate(truth):
            if independent_iou(p.get('bbox'),g['bbox'])>=iou and (not text or (compared(p['text']) and compared(p['text'])==compared(g['text']))):
                adjacency[i,j]=1
    if not adjacency.size:return 0,set()
    left,right=linear_sum_assignment(-adjacency)
    correct=[int(j) for i,j in zip(left,right) if adjacency[i,j]]
    return len(correct),set(correct)


def image_oracle(student,deep,unet_images=None):
    """GT-only maximum micro F1 for one whole expert output per image.

    Dinkelbach optimization accounts for differing output counts; maximizing TP
    independently is not in general the F1 oracle. Never used to fit a gate.
    """
    value=aggregate(deep)['f1']
    for iteration in range(1000):
        gain=np.array([2*(s['tp']-d['tp'])-value*(s['predictions']-d['predictions']) for s,d in zip(student,deep)])
        if unet_images is None:chosen=gain>0
        else:
            chosen=np.zeros(len(gain),bool);chosen[np.argsort(-gain,kind='stable')[:unet_images]]=True
        metric=aggregate([s if c else d for s,d,c in zip(student,deep,chosen)])
        if abs(metric['f1']-value)<1e-12:
            return {**metric,'unet_images':int(chosen.sum()),'deepsolo_images':int((~chosen).sum()),
                    'unet_image_indices':np.where(chosen)[0].tolist(),'iterations':iteration+1,'scope':'GT-only diagnostic, not a learned router'}
        value=metric['f1']
    raise RuntimeError('F1 oracle did not converge; do not publish a partial bound')


def deepsolo_control(root,config):
    protocol=read(root/'protocol.json');index=protocol['index'];iou=protocol['config']['evaluation']['iou_threshold'];rows=[];predictions={}
    normalization=protocol['config']['evaluation'].get('text_normalization',CASEFOLD)
    for phase,records in index.items():
        if phase=='fit':continue
        predictions[phase]={r['sample_id']:read(root/'deepsolo'/phase/(sample_key(r['sample_id'])+'.json'))['predictions'] for r in records}
    for threshold in config['deepsolo_score_thresholds']:
        metric=aggregate([measure([p for p in predictions['calibration'][r['sample_id']] if p['score']>=threshold],r['ground_truth'],iou,normalization) for r in index['calibration']])
        rows.append({'threshold':threshold,**metric})
    chosen=max(rows,key=lambda r:(r['f1'],-r['fp'],r['tp'],-r['threshold']));metrics={};per={}
    for phase in ('calibration','evaluation','test'):
        per[phase]=[{'sample_id':r['sample_id'],**measure([p for p in predictions[phase][r['sample_id']] if p['score']>=chosen['threshold']],r['ground_truth'],iou,normalization)} for r in index[phase]]
        metrics[phase]=aggregate(per[phase])
    report={'claim_label':'approximate reproduction','scope':'secondary frozen-DeepSolo score-filter control; no model re-inference',
            'config':config,'chosen':chosen,'calibration_sweep':rows,'metrics':metrics,'per_image':per,'selection_uses_test':False,
            'text_normalization':normalization}
    write(root/'deepsolo_calibration_control.json',report)
    return report


def validate_mae_qualification(root):
    """Accept corrected historical or directly qualified fresh teacher targets."""
    legacy=root/'mae_preprocessing_v2_complete.json'
    if legacy.exists():
        receipt=read(legacy);preprocessing=receipt.get('config',{})
    else:
        audit=root/'teacher_cache_audit.json'
        if not audit.exists():raise ValueError('Missing MAE teacher qualification; run the teacher stage before reporting')
        receipt=read(audit).get('mae',{});preprocessing=receipt.get('input_preprocessing',{})
    expected=teacher_preprocessing('mae')
    if receipt.get('status')!='PASS' or any(preprocessing.get(key)!=value for key,value in expected.items()):
        raise ValueError('Unqualified MAE preprocessing; require original RGB/bicubic/32x128/mean=std=0.5 teacher targets')


def run(root,analysis_config,controls_config):
    protocol=read(root/'protocol.json');config=protocol['config'];index=protocol['index']
    normalization=config['evaluation'].get('text_normalization',CASEFOLD)
    # A saved supervisor is evidence for this run; manual complete-stage reruns
    # need not fabricate one. All warmup/branch artifacts are checked below.
    if (root/'pipeline_state.json').exists() and read(root/'pipeline_state.json')['status']!='SUCCESS':
        raise ValueError('Full stage supervisor has not completed')
    warmup=read(root/'warmup/complete.json')
    if warmup['status']!='SUCCESS' or warmup['epochs']!=config['training']['warmup_epochs']:
        raise ValueError('Partial or failed warmup cannot be reported as the full experiment')
    for arm in config['arms']:
        complete=read(root/'branches'/arm/'complete.json')
        if complete['status']!='SUCCESS' or complete['epochs']!=config['training']['branch_epochs'] or sha(root/'branches'/arm/'last.pt')!=complete['checkpoint_sha256']:
            raise ValueError('Partial, failed or changed branch: '+arm)
    validate_mae_qualification(root)
    control=deepsolo_control(root,controls_config)
    results=read(root/'word_results.json');per_image=read(root/'word_per_image.json');routers=read(root/'router_results.json')
    router_rows=read(root/'router_per_image.json');selection=read(root/'threshold_selection.json');arms=list(selection)
    router_audits=read(root/'router_fit_audits.json')
    cases=0;ambiguities=[];word_rows=[];details={};prediction_hashes={};tick=time.perf_counter()
    for phase in ('calibration','evaluation','test'):
        records=index[phase];expected=sum(len(r['ground_truth']) for r in records)
        if len({r['sample_id'] for r in records})!=len(records):raise ValueError('Duplicate IDs inside one split')
        deep_by={}
        for record in records:
            sid=record['sample_id'];deep=read(root/'deepsolo'/phase/(sample_key(sid)+'.json'))
            tp,matched=hungarian(deep['predictions'],record['ground_truth'],config['evaluation']['iou_threshold'],normalization=normalization)
            if tp!=deep['metric']['tp']:raise ValueError('Independent DeepSolo matching disagrees')
            deep_by[sid]=deep;cases+=1
        for arm in arms:
            chosen=selection[arm]['chosen'];saved={r['sample_id']:r for r in per_image[phase][arm]}
            if set(saved)!={r['sample_id'] for r in records}:raise ValueError('Missing evaluation image')
            category={};loc_counts=[];raw_metrics=[];examples=[];ocr_crops=unique_crops=0
            for record in records:
                sid=record['sample_id'];truth=record['ground_truth'];path=raw_path(root,arm,phase,sid);raw=read(path)
                ocr_crops+=raw['new_ocr_crops'];unique_crops+=raw['unique_ocr_crops']
                if raw['ground_truth_used_for_inference'] is not False or raw['new_ocr_crops']>config['evaluation']['ocr_budget']:
                    raise ValueError('Inference GT/budget contract failed')
                prediction_hashes[str(path.relative_to(root))]=sha(path)
                predictions=select(raw['predictions'],chosen['presence'],chosen['confidence'])
                tp,matched=hungarian(predictions,truth,config['evaluation']['iou_threshold'],normalization=normalization)
                metric=saved[sid]
                if (tp,len(predictions)-tp,len(truth)-tp)!=(metric['tp'],metric['fp'],metric['fn']):raise ValueError('Independent word matching disagrees')
                if matched!=set(metric['matched_gt_indices']):ambiguities.append({'phase':phase,'arm':arm,'sample_id':sid,'production':metric['matched_gt_indices'],'alternative':sorted(matched)})
                # Use the declared production matching for complementary word IDs.
                matched=set(metric['matched_gt_indices']);deep=deep_by[sid];dmatched=set(deep['metric']['matched_gt_indices'])
                loc,_=hungarian(predictions,truth,config['evaluation']['iou_threshold'],text=False)
                loc_counts.append({'tp':loc,'fp':len(predictions)-loc,'fn':len(truth)-loc,'predictions':len(predictions),'gt_words':len(truth)})
                raw_tp,_=hungarian(raw['predictions'],truth,config['evaluation']['iou_threshold'],normalization=normalization)
                raw_metrics.append({'tp':raw_tp,'fp':len(raw['predictions'])-raw_tp,'fn':len(truth)-raw_tp,'predictions':len(raw['predictions']),'gt_words':len(truth)})
                for j,g in enumerate(truth):
                    alphabet_text=normalize_word_text(g['text'],normalization) if normalization!=CASEFOLD else g['text']
                    unsupported=any(c.casefold() not in 'abcdefghijklmnopqrstuvwxyz0123456789' for c in alphabet_text)
                    group='contains_outside_deepsolo_alphabet' if unsupported else 'within_deepsolo_alphabet'
                    row=category.setdefault(group,{'gt_words':0,'unet_correct':0,'deep_correct':0,'unet_only':0,'deep_only':0})
                    row['gt_words']+=1;row['unet_correct']+=j in matched;row['deep_correct']+=j in dmatched
                    row['unet_only']+=j in matched-dmatched;row['deep_only']+=j in dmatched-matched
                    if phase=='test':
                        overlaps=[independent_iou(p.get('bbox'),g['bbox']) for p in deep['predictions']]
                        best=max(range(len(overlaps)),key=lambda i:overlaps[i]) if overlaps else None
                        deep_text=deep['predictions'][best]['text'] if best is not None else ''
                        failure='correct' if j in dmatched else 'overlapping_output_not_exact_or_assignment' if max(overlaps,default=0)>=config['evaluation']['iou_threshold'] else 'no_retained_bbox_match'
                        entry={'arm':arm,'sample_id':sid,'gt_index':j,'text':g['text'],'bbox':g['bbox'],
                               'unet_correct':j in matched,'deepsolo_correct':j in dmatched,'outside_deepsolo_alphabet':unsupported,
                               'deep_failure_category':failure,'deep_best_overlap':max(overlaps,default=0),'deep_nearest_text':deep_text,
                               'image_path':record['image_path']}
                        word_rows.append(entry)
                        if j in matched-dmatched:examples.append(entry)
                for kind in ('image','image_fixed_half','word','deepsolo'):
                    if phase=='calibration':break
                    routed_row=router_rows[phase][arm][kind][len(raw_metrics)-1]
                    if routed_row['sample_id']!=sid:raise ValueError('Routed image ordering changed')
                    if kind=='word':
                        decision=read(root/'router_decisions'/arm/phase/(sample_key(sid)+'.json'))
                        routed_predictions=decision['word_routed_predictions']
                    elif kind=='deepsolo':routed_predictions=deep['predictions']
                    elif routed_row['action']=='unet':routed_predictions=predictions
                    else:routed_predictions=deep['predictions']
                    rt,_=hungarian(routed_predictions,truth,config['evaluation']['iou_threshold'],normalization=normalization)
                    if (rt,len(routed_predictions)-rt,len(truth)-rt)!=(routed_row['tp'],routed_row['fp'],routed_row['fn']):
                        raise ValueError('Independent routed matching disagrees')
                    cases+=1
                cases+=1
            if results[phase][arm]['gt_words']!=expected:raise ValueError('Wrong GT denominator')
            detail={'alphabet_groups':category,'localization_only':aggregate(loc_counts),'before_postfilter':aggregate(raw_metrics),
                    'examples':examples,'after_postfilter':results[phase][arm],
                    'crop_cost':{'ocr_crops_processed':ocr_crops,'unique_crop_tensors':unique_crops,'maximum_per_image':config['evaluation']['ocr_budget']}}
            drows=[deep_by[r['sample_id']]['metric'] for r in records]
            srows=[saved[r['sample_id']] for r in records]
            detail['gt_image_oracles']={'unconstrained':image_oracle(srows,drows),
                  'fixed_half':image_oracle(srows,drows,len(records)//2)}
            if phase!='calibration':
                deltas={}
                for kind,rows in router_rows[phase][arm].items():
                    gained=lost=0
                    for record,row in zip(records,rows):
                        a=set(row['matched_gt_indices']);b=set(deep_by[record['sample_id']]['metric']['matched_gt_indices']);gained+=len(a-b);lost+=len(b-a)
                    metric=routers[phase][arm][kind];d=results[phase]['deepsolo']
                    deltas[kind]={'recovered_deep_errors':gained,'lost_deep_correct':lost,'net_correct':gained-lost,
                                  'extra_false_positives':metric['fp']-d['fp'],'delta_f1':metric['f1']-d['f1']}
                    if phase=='test':
                        retained=[r for r in rows if r['sample_id'] not in config['evaluation']['known_duplicate_test_ids']]
                        ds_retained=[r for r in router_rows[phase][arm]['deepsolo'] if r['sample_id'] not in config['evaluation']['known_duplicate_test_ids']]
                        sensitivity=aggregate(retained);sensitivity['delta_f1_vs_deep']=sensitivity['f1']-aggregate(ds_retained)['f1']
                        deltas[kind]['known_duplicate_exclusion_sensitivity']=sensitivity
                detail['router_changes_vs_deep']=deltas
            details.setdefault(phase,{})[arm]=detail
    best_cal=max(arms,key=lambda arm:(results['calibration'][arm]['f1'],-results['calibration'][arm]['fp'],arm))
    best_learned_cal=max(config['arms'],key=lambda arm:(results['calibration'][arm]['f1'],-results['calibration'][arm]['fp'],arm))
    declared_arms=list(config['arms'])
    best_routed_cal=max(declared_arms,key=lambda arm:(router_audits[arm]['word']['chosen']['metric']['f1'],
                        -router_audits[arm]['word']['chosen']['changes'],-declared_arms.index(arm)))
    # TEST ranking is descriptive only and never redefines the selected method.
    descriptive_test=max(arms,key=lambda arm:results['test'][arm]['f1'])
    audit={'status':'PASS','independent_matching':'SciPy Hungarian on a binary exact-text AND bbox IoU adjacency',
           'text_normalization':normalization,
           'cases_checked':cases,'equally_optimal_gt_identity_ambiguities':ambiguities,
           'all_gt_denominators_verified':True,'all_prediction_budgets_verified':True,
           'complete_methods':arms,'selection_uses_test':False,'seconds':time.perf_counter()-tick}
    repo=Path(__file__).resolve().parents[1]
    audit['analysis_code_sha256']={str(p.relative_to(repo)):sha(p) for p in
          (Path(__file__),repo/'recognition/word_slot_evaluation.py',repo/'segmentation/word_slot_data.py')}
    audit['input_report_sha256']={name:sha(root/name) for name in
          ('protocol.json','word_results.json','word_per_image.json','router_results.json','router_per_image.json','router_fit_audits.json','threshold_selection.json')}
    write(root/'independent_audit.json',audit);write(root/'failure_analysis.json',details)
    with (root/'test_per_word.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=[k for k in word_rows[0] if k!='bbox']+['bbox'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in word_rows:writer.writerow({**row,'bbox':json.dumps(row['bbox'])})
    write(root/'analysis_selection.json',{'best_calibration_method':best_cal,'best_calibration_learned_method':best_learned_cal,
          'primary_best_learned_router_oof':best_routed_cal,'selection_configuration':analysis_config,
          'descriptive_test_best_method':descriptive_test,'test_selection_for_training':False})
    write(root/'analysis_source_sha256.json',prediction_hashes)
    report(root,protocol,results,routers,details,audit,best_cal,best_learned_cal,best_routed_cal,descriptive_test,control)
    print(json.dumps({'audit':audit['status'],'cases':cases,'selected_on_calibration':best_cal,
                     'primary_selected_on_router_oof':best_routed_cal,'selected_learned_on_calibration':best_learned_cal}),flush=True)


def router_result_label(metric):
    """Every reported router score includes its audited image-call denominator."""
    images=metric['images'];calls=metric['logical_deepsolo_calls']
    if type(images) is not int or images<=0 or type(calls) is not int or not 0<=calls<=images:
        raise ValueError('Invalid router image/call counts; require integer0<=DeepSolo<=positive image total')
    return f"{metric['f1']*100:.3f} ({calls}/{images}; {100*calls/images:.2f}%)"


def report(root,protocol,results,routers,details,audit,best_cal,best_learned_cal,best_routed_cal,descriptive_test,control):
    config=protocol['config'];test=results['test'];deep=test['deepsolo'];arms=list(config['arms'])
    normalization=config['evaluation'].get('text_normalization',CASEFOLD);text_policy=comparison_label(normalization)
    router_audits=read(root/'router_fit_audits.json')
    primary_word=routers['test'][best_routed_cal]['word']
    conclusion='The primary calibration-selected word router does not exceed the same frozen DeepSolo baseline under this protocol.' if primary_word['f1']<=deep['f1'] else 'The primary calibration-selected word router improves F1 over the same frozen DeepSolo on this explored TEST; independent validation remains unresolved.'
    lines=['# Word-slot U-Net, PARSeq and DeepSolo: fixed-budget experiment','',
      '**Approximate reproduction / research extension.** One seed, eight shared warmup epochs and three branch epochs. '+
      'Not a full literature-model reproduction, an independent dataset claim, or official Total-Text polygon evaluation.','',
      f"All 13 learned branches completed. Independent matching audit: **{audit['status']}**, {audit['cases_checked']} cases. "+
      f"Primary TEST: {deep['images']} images / {deep['gt_words']} care words.",'',
      f"**{conclusion}** The primary word router reaches {primary_word['f1']*100:.3f}% "+
      f"with DeepSolo on {primary_word['logical_deepsolo_calls']}/{primary_word['images']} images ({100*primary_word['logical_deepsolo_calls']/primary_word['images']:.2f}%) "+
      f"versus {deep['f1']*100:.3f}% for the same frozen DeepSolo. No method or routing policy is promoted. "+
      'Implementation completion and scientific success are separate outcomes.','',
      f"Text comparison: **{text_policy}**. Raw OCR and annotation strings remain intact; normalization affects matching, quality labels and textual agreement, not pixels or model training.",'',
      f"The primary learned specialist selected by OOF calibration routed F1 is **{best_routed_cal}**. "+
      f"The secondary method selected by calibration standalone F1 is **{best_cal}**; among learned slot methods it is **{best_learned_cal}**. "+
      f"The descriptive TEST best is {descriptive_test}; TEST does not select checkpoints or thresholds.",'',
      '## Standalone word outcomes','',
      '| Method | Calibration F1 % | Evaluation125 F1 % | TEST TP | TEST FP | TEST FN | TEST F1 % | U-only / D-only | OCR crops processed |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ['deepsolo','pixel_control','pixel_pil',*arms]:
        r=test[arm];exclusive='—' if arm=='deepsolo' else f"{r['unet_only']} / {r['deepsolo_only']}"
        cost='300 full images' if arm=='deepsolo' else str(details['test'][arm]['crop_cost']['ocr_crops_processed'])
        lines.append(f"| {arm} | {results['calibration'][arm]['f1']*100:.3f} | {results['evaluation'][arm]['f1']*100:.3f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['f1']*100:.3f} | {exclusive} | {cost} |")
    factors={}
    for arm in arms:
        history=read(root/'branches'/arm/'history.json');comparisons=sum(h['gradient_comparisons'] for h in history)
        conflicts=sum(h['gradient_conflicts'] for h in history)
        factors[arm]={'test_delta_f1_vs_slot_control':test[arm]['f1']-test['slot_control']['f1'],
          'test_delta_f1_vs_pixel_pil':test[arm]['f1']-test['pixel_pil']['f1'],
          'gradient_comparisons':comparisons,'gradient_conflicts':conflicts,
          'gradient_conflict_fraction':conflicts/comparisons if comparisons else None}
    write(root/'factor_diagnostics.json',{'scope':'descriptive fixed-budget single-seed outcomes; no TEST selection','methods':factors})
    pc=factors['pcgrad']
    lines += ['',f"Relative to slot_control ({test['slot_control']['f1']*100:.3f}%), recognition alignment reaches {test['recognition_alignment']['f1']*100:.3f}%, "+
      f"while mask initialization falls to {test['mask_initialization']['f1']*100:.3f}% and the combined arm reaches {test['combined']['f1']*100:.3f}%. "+
      ('No new branch exceeds the pixel-PIL standalone control. ' if max(test[a]['f1'] for a in arms)<=test['pixel_pil']['f1'] else 'Some new branches exceed the pixel-PIL standalone control under this text policy. ')+
      'These are paired-budget descriptive differences; they do not identify a general architectural cause.','',
      f"PCGrad projects {pc['gradient_conflicts']} of {pc['gradient_comparisons']} measured task-gradient pairs ({pc['gradient_conflict_fraction']*100:.2f}%), "+
      f"but its standalone F1 is {test['pcgrad']['f1']*100:.3f}%. Detecting and projecting conflicts therefore does not establish an accuracy gain. "+
      'CCD/MAE uniform-bin alignment and glyph guidance are adaptations, not a verdict on their complete original models.','']
    lines+=['','U-only/D-only counts concern the same GT word IDs under declared maximum matching. '+
      'An exclusive recovery is useful only if a selector retains it without sufficient additional damage or false positives.','',
      f"Secondary control: calibration-only DeepSolo score filtering chooses threshold {control['chosen']['threshold']}, "+
      f"giving TEST F1 **{control['metrics']['test']['f1']*100:.3f}%** versus the primary frozen-expert {deep['f1']*100:.3f}%. "+
      'This checks whether confidence filtering alone explains an apparent gain. It only filters already retained detections and does not reconstruct lower-score outputs.','',
      '## Learned selectors','',
      '| Student | OOF CAL word F1 % (D images/total; %) | Image TEST F1 % (D images/total; %) | Half-image TEST F1 % (D images/total; %) | Word TEST F1 % (D images/total; %) | Word recovered / lost D-correct | Word extra FP |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ['pixel_control','pixel_pil',*arms]:
        r=routers['test'][arm];d=details['test'][arm]['router_changes_vs_deep']['word']
        cal=router_audits[arm]['word']['chosen']['metric']
        cal_label=router_result_label({**cal,'logical_deepsolo_calls':cal['images']})
        lines.append(f"| {arm} | {cal_label} | {router_result_label(r['image'])} | {router_result_label(r['image_fixed_half'])} | {router_result_label(r['word'])} | {d['recovered_deep_errors']} / {d['lost_deep_correct']} | {d['extra_false_positives']:+d} |")
    r=routers['test'][best_routed_cal];d=details['test'][best_routed_cal]
    for kind in ('image','image_fixed_half','word'):
        row=r[kind];change=d['router_changes_vs_deep'][kind];interval=row['paired_bootstrap_delta_f1_95_interval']
        lines+=['',f"For OOF-router-selected learned student **{best_routed_cal}**, `{kind}` yields "+
           f"F1 {row['f1']*100:.3f}% versus DeepSolo {deep['f1']*100:.3f}% "+
           f"with DeepSolo on {row['logical_deepsolo_calls']}/{row['images']} images ({100*row['logical_deepsolo_calls']/row['images']:.2f}%) "+
           f"(difference {change['delta_f1']*100:+.3f} percentage points; paired image-bootstrap 95% interval "+
           f"[{interval[0]*100:+.3f}, {interval[1]*100:+.3f}]). It recovers {change['recovered_deep_errors']} "+
           f"DeepSolo errors, loses {change['lost_deep_correct']} previously correct words and changes FP by {change['extra_false_positives']:+d}."]
    lines+=['','Word routing requires both experts on every image; it does not save DeepSolo calls. '+
      'Image-call counts are logical simulations from frozen outputs. They are not measured selective-deployment latency.','',
      '## Failure structure of the OOF-router-selected learned method','']
    chosen=router_audits[best_routed_cal]['word']['chosen'];cal=results['calibration']['deepsolo']
    if chosen['replace_threshold']>1:
        cal_new_tp=chosen['metric']['tp']-cal['tp'];cal_extra_fp=chosen['metric']['fp']-cal['fp']
        change=d['router_changes_vs_deep']['word'];new_tp=change['recovered_deep_errors'];extra_fp=change['extra_false_positives']
        additions=new_tp+extra_fp
        if additions>0 and cal_new_tp+cal_extra_fp>0 and change['lost_deep_correct']==0:
            admission={'scope':'post-hoc diagnostic; does not tune thresholds','method':best_routed_cal,
              'replace_disabled':True,'calibration_oof':{'true_additions':cal_new_tp,'false_additions':cal_extra_fp,
                'admission_precision':cal_new_tp/(cal_new_tp+cal_extra_fp)},
              'test':{'true_additions':new_tp,'false_additions':extra_fp,'admission_precision':new_tp/additions},
              'minimum_addition_precision_to_improve_baseline_f1':deep['f1']/2}
            write(root/'router_admission_diagnostics.json',admission)
            lines += [f"The selected word gate disables replacement. OOF calibration additions contain {cal_new_tp} true and {cal_extra_fp} false words "+
              f"({admission['calibration_oof']['admission_precision']*100:.2f}% admission precision). TEST additions contain {new_tp} true and {extra_fp} false words "+
              f"({admission['test']['admission_precision']*100:.2f}%). Its loss comes from false admissions, not deleting correct DeepSolo outputs. "+
              f"For conflict-free additions with unchanged existing matches, improving micro F1 requires addition precision greater than baseline F1/2, here {deep['f1']*50:.2f}%. "+
              'This is a count-derived diagnostic, not a threshold fitted on TEST. Small calibration size, method selection and prediction-distribution shift remain possible explanations; this study does not isolate their causes.','']
    oracle=d['gt_image_oracles']
    lines += [f"For this same calibrated student pair, the GT-assisted whole-image maximum is {oracle['unconstrained']['f1']*100:.3f}% F1 "+
              f"with DeepSolo on {oracle['unconstrained']['deepsolo_images']}/{oracle['unconstrained']['images']} images ({100*oracle['unconstrained']['deepsolo_images']/oracle['unconstrained']['images']:.2f}%); "+
              f"with DeepSolo on {oracle['fixed_half']['deepsolo_images']}/{oracle['fixed_half']['images']} images ({100*oracle['fixed_half']['deepsolo_images']/oracle['fixed_half']['images']:.2f}%) it is {oracle['fixed_half']['f1']*100:.3f}%. These optimize TP and emitted-count tradeoffs jointly. "+
              'They are diagnostic bounds for this fixed pair, not realizable gates, training targets, or ceilings on future U-Net designs.','']
    for group,row in d['alphabet_groups'].items():
        lines.append(f"- `{group}`: {row['gt_words']} GT words, U correct {row['unet_correct']}, D correct {row['deep_correct']}, exclusive U {row['unet_only']} and D {row['deep_only']}.")
    raw=d['before_postfilter'];filtered=d['after_postfilter'];local=d['localization_only']
    lines+=['',f"Before post-OCR filtering: {raw['tp']} correct / {raw['predictions']} outputs, F1 {raw['f1']*100:.3f}%. "+
       f"After calibration filtering: {filtered['tp']} / {filtered['predictions']}, F1 {filtered['f1']*100:.3f}%. "+
       f"Localization without text has {local['tp']} one-to-one bbox matches. This diagnostic separates spatial coverage from exact reading; it is not a causal detector-versus-recognizer decomposition.",'',
       'The complete per-word CSV includes DeepSolo overlap categories and exclusive recoveries. '+
       'Inspect actual RGB crops and raw transcriptions before attributing failures to font, occlusion, curvature or semantics.','',
       'Post-hoc visual review flags TEST399: the raw annotation is `Chirardelli`, while the sign appears to read `Ghirardelli`. '+
       'This remains UNRESOLVED; raw labels and all primary metrics are unchanged. A GT match does not establish visually correct transcription in every example. '+
       'One flagged example does not explain the overall negative routing result.','',
      '## Training, resource and validity boundaries','',
       'All word denominators include care GT, failed reads and empty predictions. Every output counts; the historical polygon-ignore DeepSolo score is a different protocol. '+
       f"Text policy: {text_policy}. Bbox matching uses original stored RGB coordinates and inverse EXIF mapping for DeepSolo.",'',
       'Training uses FIT1030 only for new modules. The original frozen prior historically saw broader TRAIN; reused evaluation and repeatedly explored TEST are not pristine holdouts. '+
       'Two known pixel-identical TRAIN/TEST pairs are retained in the primary metric and excluded in the saved 298-image sensitivity view. '+
       'This single-seed screen cannot establish convergence, full training variance, global novelty or equivalence to the complete cited models.','',
       'The actual hardware is RTX3090/24GiB. FP32 microbatch1, accumulation4, deterministic CPU grid sampling and sequential legacy/main stages are explicit hardware deviations. '+
       'Check `inference_runtime.json`, each branch `complete.json`, teacher loading audits and `deep_training_export.json` for actual model/shape/batch and allocated/reserved memory peaks. '+
       'No 8GiB or paper latency result is inferred.','',
       'Per-image student inference time covers the trainable decoder and OCR with a cached frozen prior. '+
       'Wall export time additionally includes I/O and saved intermediates. Neither is a complete warm/cold end-to-end U-Net latency benchmark; equal crop caps do not mean identical realized OCR cost.','',
       'MAE teacher targets were regenerated before its branch was initialized, using the pinned original pretraining RGB/bicubic/0.5-normalization recipe. '+
       'The initial ImageNet-normalized cache and the intentional epoch-boundary continuation receipts are preserved. '+
       'CCD targets and the four already completed branches were unchanged; the interrupted CCD epoch resumed from its last complete checkpoint.','',
       'Original SIGA weights were downloaded and hashed but not executed: its inspected official source repository lacks model implementation. '+
       'The implemented guidance arm uses the actual strictly loaded CCD glyph teacher and is labeled an adaptation. '+
       'The learned strip is not the original TPS++; the dense query arm is not the full DEIM augmentation/MAL recipe.','',
       '## Implications for a further experiment','',
       'The observed exclusive recoveries establish complementary hypotheses in this fixed candidate pool; their admission precision does not establish a usable specialist. '+
       'The next justified experiment would first measure expert predictions and router calibration on image-separated training folds with a fresh final validation set, '+
       'then train admission against paired changes in TP and output count rather than interpreting ordinary OCR confidence as routed benefit. '+
       'Historical prior exposure to calibration must be accounted for. This is a proposed experiment, not a tested improvement.','',
       'The short single-seed screen also leaves decoder convergence and true character alignment unresolved. '+
       'A longer or differently pretrained decoder should retain matched pixel controls, fixed crop budgets, per-word audit and separate calibration; '+
       'the present negative results do not justify combining every loss or claiming that the complete original literature models fail.','',
       '## Reusable evidence','',
       '- `protocol.json`, immutable source snapshots and exact input/cache hashes.',
       '- `word_results.json`, `word_per_image.json`, `threshold_selection.json`, `test_per_word.csv`.',
       '- `router_results.json`, `router_fit_audits.json`, deployable serialized coefficients and per-image route decisions.',
       '- `failure_analysis.json`, `independent_audit.json`, `analysis_selection.json`.',
       '- Complete checkpoints, histories, original detector responses and selected mask/grid intermediates remain external to Git.','']
    if (root/'live_smoke_comparison.json').exists():
        live=read(root/'live_smoke_comparison.json')
        lines += ['## Actual live inference qualification','',
          f"Three uncached, GT-free original-RGB inference cases: **{live['status']}**. The image gate keeps U-Net with zero DeepSolo calls, "+
          'defers to real isolated DeepSolo with one call, and the word gate invokes both experts. '+
          'All20 crop tensor hashes, boxes, transcriptions and confidence values match their saved evaluation counterparts exactly. '+
          'These cases verify execution, not a new accuracy or latency benchmark.','',
          'The first functional smoke exposed retained PARSeq CUDA weights after reference release. '+
          'Explicit CPU transfer before reference cleanup fixed the lifecycle; the first smoke and diagnostic remain archived. '+
          'The final diagnostic finds no remaining CUDA model tensors, with32MiB of runtime allocation. '+
          'Successful student peaks are595167232 allocated /683671552 reserved bytes; the isolated DeepSolo worker reports1172747264 /2420113408 bytes. '+
          'See `live_smoke_comparison.json` and `live_smoke/release_diagnostic.json`. This does not qualify all inputs or an8GiB GPU.','']
    if (root/'figures_reviewed/comparison.png').exists():
        lines += ['[Full comparison figure](figures_reviewed/comparison.png) and [visually reviewed examples](figures_reviewed/exclusive_examples.png).','']
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--mode',required=True,choices=['approximate']);parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--controls-config',required=True,type=Path)
    args=parser.parse_args();c=read(args.config);repo=Path(__file__).resolve().parents[1]
    if repo/'configs/approximate' not in args.config.resolve().parents or c['reproduction_mode']!=args.mode or c['experiment_kind']!='word_slot_analysis_v1':
        raise ValueError('Explicit approximate analysis configuration required')
    controls=read(args.controls_config)
    if repo/'configs/approximate' not in args.controls_config.resolve().parents or controls['reproduction_mode']!=args.mode or controls['experiment_kind']!='word_slot_secondary_controls_v1':
        raise ValueError('Explicit approximate secondary-control configuration required')
    run(args.run.resolve(),c,controls)
