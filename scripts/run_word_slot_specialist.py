"""Run a completed specialist and a fitted image/word router on a new RGB file."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from deepsolo.adapter import DeepSoloSubprocessAdapter
from pipeline.word_slot_specialist import WordSlotSpecialist
from recognition.word_slot_routing import FrozenWordSlotRouter
from recognition.clip_router_evaluation import deepsolo_predictions
from scripts.run_word_slot_methods import setup
from segmentation.word_slot_data import read,write,sha

REPO=Path(__file__).resolve().parents[1]


def run(args):
    root=args.run.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError('Use a fresh live output directory')
    output.mkdir(parents=True);protocol=read(root/'protocol.json');config=protocol['config']
    if args.mode!='approximate' or config['reproduction_mode']!=args.mode:raise ValueError('Explicit APPROXIMATE completed run required')
    arm=args.arm or read(root/'analysis_selection.json')['primary_best_learned_router_oof']
    hardware=setup(config);write(output/'runtime.json',hardware)
    specialist=WordSlotSpecialist(root,arm,mode=args.mode);raw=specialist.predict(args.image,artifact_directory=output)
    write(output/'student.json',raw)
    fit=read(root/'router_fit_audits.json')[arm];router_config=read(root/'router_execution.json')['config'];router=FrozenWordSlotRouter(fit,router_config)
    release_trace=[];deep_trace=[]
    def release():release_trace.append(specialist.release())
    def fallback():
        if sha(args.image)!=raw['input']['sha256']:raise ValueError('RGB bytes changed before fallback')
        plan=read(config['paths']['deepsolo_plan']);resolution=plan['candidate_resolution']
        source=resolution['runtime_metadata']['source_identity_verification']['host_source_root']
        adapter=DeepSoloSubprocessAdapter(Path(config['paths']['deepsolo_python']).absolute(),Path(config['paths']['deepsolo_runtime']),
                                          host_source_root=Path(source),repository_root=REPO)
        response=adapter.infer(image_path=args.image.resolve(),candidate_id=resolution['candidate']['artifact_id'],
                    inference_options={'device':'cuda:0','score_threshold':plan['config']['score_threshold'],'lexicon_type':None},
                    reduction_policy='reading_order_space_join').to_dict()
        write(output/'deepsolo_response.json',response);deep_trace.append({'response_sha256':sha(output/'deepsolo_response.json'),'memory':response['memory'],'latency':response['latency']})
        predictions=deepsolo_predictions(response,raw['input']['image_wh'],raw['input']['orientation'],np.ones((128,128),bool),plan['config']['score_threshold'])
        for p in predictions:p['ignored']=False;p['confidence']=p['score'];p['presence']=1.
        return predictions
    try:
        start=time.perf_counter();result=router.route(raw['predictions'],raw['input']['image_wh'],kind=args.router_kind,
                                                    release_unet=release,run_deepsolo=fallback)
        write(output/'result.json',{'status':'SUCCESS','schema_version':1,'mode':args.mode,'claim_label':'approximate reproduction',
                   'result_class':'single-image live smoke / research inference','input':raw['input'],'arm':arm,'router_kind':args.router_kind,
                   'routing':result,'release_trace':release_trace,'deep_trace':deep_trace,'routing_seconds':time.perf_counter()-start,
                   'ground_truth_used':False,'cached_prior_used':False,'student_trace_sha256':sha(output/'student.json'),
                   'protocol_sha256':sha(root/'protocol.json'),'router_fit_sha256':sha(root/'router_fit_audits.json'),
                   'router_execution_sha256':sha(root/'router_execution.json'),'arm_explicitly_overridden':args.arm is not None,
                   'code_sha256':{str(p.relative_to(REPO)):sha(p) for p in (Path(__file__),REPO/'pipeline/word_slot_specialist.py',REPO/'recognition/word_slot_routing.py')}})
        print({'status':'SUCCESS','arm':arm,'router_kind':args.router_kind,'expert':result['expert'],'deepsolo_calls':result['deepsolo_calls']},flush=True)
    finally:specialist.release()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',required=True,choices=['approximate'])
    parser.add_argument('--run',required=True,type=Path);parser.add_argument('--image',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--arm');parser.add_argument('--router-kind',required=True,choices=['image','word'])
    args=parser.parse_args()
    try:run(args)
    except Exception as error:
        if args.output.exists():write(args.output/('failure_'+str(time.time_ns())+'.json'),{'status':'FAILED','type':type(error).__name__,
                    'message':str(error),'traceback':traceback.format_exc(),'oom':isinstance(error,torch.cuda.OutOfMemoryError)})
        raise
