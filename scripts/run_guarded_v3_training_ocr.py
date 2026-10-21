"""Durable sequential V3 training -> selected-checkpoint PARSeq -> independent audit."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import traceback

from segmentation.data import file_sha256

REPO=Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def train_command(c, check=False):
    root=Path(c['run_root'])
    command=[c['python'],'-u','-m','scripts.train_guarded_v3','--mode','approximate',
        '--config',str(REPO/c['training_config']),'--data-root',c['training_data_root'],
        '--encoder-checkpoint',c['encoder_checkpoint'],'--output-root',str(root/'training'),
        '--device',c['runtime']['device'],'--check-report',str(root/('preflight_training.json' if check else 'training_failure.json'))]
    return command+(['--check'] if check else [])


def resolved_ocr_plan(c, run, area):
    source=json.loads((run/'run_manifest.json').read_text())
    if not source['training_complete'] or source['completed_epochs']!=source['configured_epochs'] or source['test_loaded']:
        raise ValueError('OCR must wait for complete TRAIN-only checkpoint selection')
    selected=json.loads((run/'checkpoint_selection.json').read_text())
    plan=copy.deepcopy(c['ocr_template'])
    plan.update(checkpoint_epochs=sorted({r['epoch'] for r in selected.values()}),
        min_unpadded_bbox_area_native=area,
        source_sha256={name:file_sha256(run/name) for name in ['run_manifest.json','run_contract.json','epochs.jsonl','checkpoint_selection.json']})
    if not 1<=len(plan['checkpoint_epochs'])<=4:raise ValueError('Unexpected checkpoint population')
    return plan


def parseq_smoke(c, root):
    import numpy as np
    from PIL import Image
    import torch
    from recognition.parseq import ParseqApproxConfig,ParseqApproxRecognizer,PARSEQ_ARCHIVE_SOURCE
    torch.set_num_threads(c['runtime']['cpu_threads']);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(c['parseq_source']),
        checkpoint=Path(c['parseq_checkpoint']),device=c['runtime']['device'],runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
        runtime_source_archive=Path(c['parseq_source_archive'])))
    try:
        with Image.open(c['smoke_image']) as im:rgb=np.asarray(im.convert('RGB')).copy()
        values=recognizer.recognize_candidate_batch([rgb])
        if len(values)!=1:raise RuntimeError('PARSeq output cardinality differs')
        report={'status':'PASS','result_class':'smoke-test result','claim_label':'approximate reproduction',
            'device':c['runtime']['device'],'precision':'float32','model_set':['qualified PARSeq AR/refine1'],
            'input_source_sha256':file_sha256(Path(c['smoke_image'])),'raw_text':values[0].raw_text,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
            'batch_size':1,'recognizer_input_shape':[1,3,32,128],'out_of_memory':False,'saved_crops':0}
        write(root/'preflight_parseq.json',report)
    finally:recognizer.release()


def run(args):
    c=json.loads(args.config.read_text());root=Path(c['run_root']);root.mkdir(parents=True,exist_ok=True)
    if args.config.resolve().parent!=REPO/'configs/approximate' or c['variant']!='guarded-v3-training-then-ocr-v1' or c['reproduction_mode']!='approximate':
        raise ValueError('Explicit isolated APPROXIMATE V3 pipeline required')
    env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=c['runtime']['cublas_workspace_config']
    os.environ['CUBLAS_WORKSPACE_CONFIG']=env['CUBLAS_WORKSPACE_CONFIG']
    def child(command, log):
        with log.open('a') as stream:
            stream.write(json.dumps({'command':command,'utc':datetime.now(timezone.utc).isoformat()})+'\n');stream.flush()
            subprocess.run(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    if args.check:
        child(train_command(c,True),root/'preflight_training.log')
        parseq_smoke(c,root)
        write(root/'preflight.json',{'status':'PASS','claim_label':'approximate reproduction',
            'pipeline_config_sha256':file_sha256(args.config),'new_full_training':False})
        return
    with (root/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'status':'RUNNING','phase':'TRAINING','claim_label':'approximate reproduction',
            'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),'config_sha256':file_sha256(args.config),
            'frozen_code_root':str(REPO),'saved_crops':0}
        preflight=json.loads((root/'preflight.json').read_text())
        if preflight['status']!='PASS' or preflight['pipeline_config_sha256']!=file_sha256(args.config):
            raise ValueError('GPU preflight for this exact pipeline configuration required')
        write(root/'pipeline_status.json',state)
        try:
            child(train_command(c),root/'training.log')
            training=json.loads((REPO/c['training_config']).read_text());run=root/'training'/training['run_name']
            rows=[]
            for area in c['area_filters']:
                state.update(phase='PARSEQ',area_filter=area);write(root/'pipeline_status.json',state)
                config=REPO/'configs/approximate'/f'resolved_guarded_v3_ocr_area{area}.json'
                write(config,resolved_ocr_plan(c,run,area));output=root/'ocr'/f'area{area}'
                command=[c['python'],'-u','-m','scripts.evaluate_totaltext_compact_v2_selected_ocr',
                    '--config',str(config),'--run-directory',str(run),'--test-manifest',c['test_manifest'],
                    '--data-root',c['test_data_root'],'--parseq-source',c['parseq_source'],
                    '--parseq-source-archive',c['parseq_source_archive'],'--parseq-checkpoint',c['parseq_checkpoint'],
                    '--output-dir',str(output)]
                child(command,root/f'ocr_area{area}.log')
                child([c['python'],'-u','-m','scripts.audit_totaltext_compact_v2_selected_ocr','--result-directory',str(output)],root/f'ocr_area{area}_audit.log')
                measured=json.loads((output/'results.json').read_text());write(output/'resolved_plan.json',json.loads(config.read_text()))
                for epoch,checkpoint in measured['checkpoints'].items():
                    for pad,summary in checkpoint['summary'].items():
                        for mode in ['unpadded_detection_box','padded_crop_box']:
                            m=summary[mode];g=m['geometry'];n=g['retained_proposals'];gt=g['gt_instances'];tp=m['recognition']['exact_casefold_spatial_one_to_one']
                            rows.append({'epoch':int(epoch),'bbox_min_area':area,'padding_native':int(pad),'box_metric':mode,
                                'images':measured['images_per_checkpoint'],'gt_words':gt,'proposals':n,
                                'localized_words':g['iou_ge_0_50_one_to_one'],'correct_words':tp,
                                'ocr_precision':tp/n if n else 0,'ocr_recall':tp/gt if gt else 0,'ocr_f1':2*tp/(n+gt) if n+gt else 0})
            with (root/'ocr_summary.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            write(root/'ocr_summary.json',{'status':'SUCCESS','claim_label':'approximate reproduction','results':rows,
                'note':'All checkpoints chosen on TRAIN validation only; TEST padding sweep is exploratory, no reranking feeds training.'})
            state.update(status='SUCCESS',phase='COMPLETE',completed_utc=datetime.now(timezone.utc).isoformat())
            write(root/'pipeline_status.json',state)
        except Exception as error:
            state.update(status='FAILED',error=str(error),traceback=traceback.format_exc())
            write(root/'pipeline_status.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--check',action='store_true');run(p.parse_args())
