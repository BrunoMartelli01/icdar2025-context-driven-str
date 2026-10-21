"""Complete missing frozen TRAIN DeepSolo branches without refitting any gate."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

from deepsolo.adapter import DeepSoloSubprocessAdapter
from recognition.clip_router_evaluation import deepsolo_predictions, evaluate_predictions
from recognition.clip_router_threshold_sweep import validate_population
from recognition.paper_image_router_evaluation import word_metrics
from scripts import run_clip_router_e2e as source
from scripts import run_clip_router_prompt_ablation as frozen
from scripts.crop_clip_runtime import sha, write_json
from scripts.run_clip_high_recall_admission import payload_digest, seal
from scripts.run_clip_specialized_routers import CONFIG, PATHS, REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['approximate'])
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--paths',type=Path,default=PATHS)
    parser.add_argument('--source-config',type=Path,default=REPO/'configs/approximate/clip_router_e2e_v1.json')
    parser.add_argument('--source-paths',type=Path,default=REPO/'configs/approximate/clip_router_e2e_local_paths_v1.json')
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.is_relative_to(REPO):
        parser.error('Use an isolated external output directory')
    args.output_root = None
    cfg, paths, observed, geometric, route_hashes = frozen.load_protocol(args)
    if sha(Path(paths['word_branches'])) != cfg['word_branch_sha256']:
        raise ValueError('Frozen word branch changed')
    word = json.loads(Path(paths['word_branches']).read_text())
    for name,digest in word['input_sha256'].items():
        if sha(Path(name)) != digest:
            raise ValueError('Word evaluation source changed')
    values = source.qualify(SimpleNamespace(mode='approximate',config=args.source_config,
        paths=args.source_paths,output_root=None))
    source_cfg, _, _, _, _, _, _, fallback, fallback_paths, _, _, splits = values
    samples = {r['observed']['sample_id']:r for r in splits['train']}
    ids = [r['sample_id'] for r in word['splits']['train']]
    validate_population(word['splits']['train'],ids,cfg['gt_words']['train'])
    if set(samples) != set(ids) or ids != [r['sample_id'] for r in observed['train']]:
        raise ValueError('Frozen TRAIN population differs')
    for row in word['splits']['train']:
        sample = samples[row['sample_id']]
        if sample['ground_truth'] != row['ground_truth']:
            raise ValueError('Word evaluation GT differs from qualified geometric source')
    pending = [r['sample_id'] for r in word['splits']['train'] if r['deepsolo'] is None]
    code = (*source.CODE,'recognition/paper_image_router_evaluation.py',
        'scripts/complete_clip_router_train_fallback.py')
    plan = {'reproduction_mode':'approximate','claim_label':'approximate reproduction',
        'result_class':'approximate reproduction','table_5_claim':False,
        'purpose':'supplement all missing TRAIN fallback outputs; no router refitting',
        'specialized_config':cfg,'source_config':source_cfg,'fallback_config':fallback,
        'fallback_paths':fallback_paths,'source_word_branches':str(paths['word_branches']),
        'source_word_branch_sha256':cfg['word_branch_sha256'],
        'source_geometric_branch_sha256':cfg['branch_metrics_sha256'],
        'train_ids':ids,'pending_train_ids':pending,'route_sha256':route_hashes,
        'code_sha256':{p:sha(REPO/p) for p in code},
        'scientific_choices':[
            {'classification':'PAPER_SPECIFIED','choice':'Algorithm2 uses full-image DeepSolo for uncertain outputs.'},
            {'classification':'REPOSITORY_SPECIFIED','choice':'Reuse qualified ResNet50, threshold0.4, no lexicon, native polygon ignore and EXIF handling, full raw detections and frozen word/geometric evaluators.'},
            {'classification':'ASSUMPTION','choice':'Run every previously uncached TRAIN image to permit comparison of arbitrary frozen gates. Retain old branches and TEST unchanged. No GT affects inference or gate selection.'},
            {'classification':'HARDWARE_DEVIATION','choice':'Existing isolated FP32 Python3.8 CUDA subprocess, one image at a time after CLIP has exited, on RTX3090. Preserve per-image model/memory/OOM/failure receipts.'},
            {'classification':'UNRESOLVED','choice':'Exact paper checkpoint and recognition-accuracy definition remain unresolved; this is no strict/Table5 claim.'}]}
    root = output/hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()[:12]
    if args.check:
        print(json.dumps({'status':'PREFLIGHT_PASS','train_images':len(ids),
            'new_deepsolo_requests':len(pending),'output':str(root)}),flush=True)
        return
    seal(root/'resolved_configuration.json',plan)
    for name,digest in plan['code_sha256'].items():
        content = (REPO/name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError('Source changed while taking experiment snapshot')
        target = root/'code_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha(target) != digest:
            raise ValueError('Source snapshot changed')
        if not target.exists():
            target.write_bytes(content)
    completed = {}
    adapter = DeepSoloSubprocessAdapter(Path(fallback_paths['deepsolo_python']).absolute(),
        source.resolve(fallback_paths['deepsolo_runtime_manifest']),
        host_source_root=source.resolve(fallback_paths['deepsolo_source_root']),repository_root=REPO)
    try:
        for index,sid in enumerate(pending,1):
            sample = samples[sid];observed_image = sample['observed']
            path = root/'deepsolo'/(source.key(sid)+'.json')
            contract = {'plan_sha256':sha(root/'resolved_configuration.json'),
                'sample_id':sid,'image_sha256':observed_image['image_sha256']}
            if path.exists():
                record = json.loads(path.read_text())
                if record['payload_sha256'] != payload_digest(record) or record['contract'] != contract:
                    raise ValueError('Supplemented DeepSolo cache changed')
            else:
                response = adapter.infer(image_path=Path(observed_image['image_path']),
                    candidate_id=fallback['candidate'],inference_options={
                        'device':fallback['device'],'score_threshold':fallback['score_threshold'],'lexicon_type':None},
                    reduction_policy='reading_order_space_join').to_dict()
                record = source.save_shard(path,contract,{'response':response})
            response = source.validate_deepsolo(record['response'],fallback,fallback['score_threshold'])
            predictions = deepsolo_predictions(response,sample['image_wh'],sample['orientation'],
                sample['validity'],fallback['score_threshold'])
            completed[sid] = {'word':word_metrics(predictions,sample['ground_truth']),
                'geometric':evaluate_predictions(predictions,sample['ground_truth'],iou_threshold=source_cfg['matching_iou']),
                'predictions':predictions,'response_path':str(path),'response_sha256':sha(path)}
            print(f'[supplement TRAIN DeepSolo] {index}/{len(pending)} {sid}',flush=True)
        supplemented = copy.deepcopy(word)
        supplemented_geometry = copy.deepcopy(geometric)
        for data,key in ((supplemented['splits'],'word'),(supplemented_geometry,'geometric')):
            for row in data['train']:
                if row['sample_id'] in completed:
                    if row['deepsolo'] is not None:
                        raise ValueError('Cannot replace an existing frozen DeepSolo branch')
                    row['deepsolo'] = completed[row['sample_id']][key]
            for split in ('train','test'):
                validate_population(data[split],[r['sample_id'] for r in observed[split]],cfg['gt_words'][split])
                if any(r['deepsolo'] is None for r in data[split]):
                    raise ValueError('Supplement still has unavailable DeepSolo outputs')
        supplemented['input_sha256'].update({v['response_path']:v['response_sha256'] for v in completed.values()})
        supplemented['input_sha256'][str(root/'resolved_configuration.json')] = sha(root/'resolved_configuration.json')
        seal(root/'completed_word_branches.json',supplemented)
        seal(root/'completed_geometric_branches.json',{'splits':supplemented_geometry})
        seal(root/'completion.json',{'status':'SUCCESS','claim_label':'approximate reproduction',
            'reproduction_mode':'approximate','result_class':'approximate reproduction',
            'table_5_claim':False,'plan_sha256':sha(root/'resolved_configuration.json'),
            'new_train_inferences':len(completed),'train_images':len(ids),'test_images':len(observed['test']),
            'source_word_branch_sha256':cfg['word_branch_sha256'],'per_image':completed,
            'word_branch_sha256':sha(root/'completed_word_branches.json'),
            'geometric_branch_sha256':sha(root/'completed_geometric_branches.json')})
        write_json(output/'latest.json',{'completion_path':str(root/'completion.json'),
            'completion_sha256':sha(root/'completion.json')})
        print(json.dumps({'status':'SUCCESS','completion_path':str(root/'completion.json')}),flush=True)
    except Exception as error:
        write_json(root/'failure.json',{'status':'FAILED','completed':len(completed),
            'error':str(error),'traceback':traceback.format_exc(),
            'worker_response':error.response.to_dict() if getattr(error,'response',None) is not None else None})
        raise


if __name__ == '__main__':
    main()
