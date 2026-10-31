"""Controlled fragment GNN and clean-to-occluded U-Net pilots on explicit CUDA."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from localize.graph_occlusion_research import (FragmentGraph, fragment_labels, neighbor_graph,
    graph_supervision, graph_groups, balanced_binary_loss, occlude_instances,
    teacher_reliability, pixel_distillation, instance_structure_distillation)
from localize.occlusion_four_tests import (decode_instance_labels, row, features, ocr_features, iou)
from localize.teacher_student_tests import CandidateScorer, compatible_selection
from scripts.run_teacher_student_tests import (source_model, source_records, frozen_selector,
    report, baseline_predictions, supervised_loss, export_model, release, peaks)
from scripts.run_occlusion_four_tests import load_rgb, path_for, labels_for, score_record, preselect, changes
from scripts.run_mask_crop_budget import read, write, utc, setup, CropOCR
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _tensor
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from scripts.evaluate_cached_totaltext_padding import crop_hash
from segmentation.data import file_sha256, load_segmentation_manifest, SegmentationDataset
from segmentation.totaltext_word_kernel import WordKernelDataset
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask, remap_letterbox_native_box
from segmentation.training import seed_everything

REPO = Path(__file__).resolve().parents[1]


def announce(out, stage, **values):
    write(out / 'status.json', dict(status='RUNNING', phase=stage, utc=utc(), pid=os.getpid(), **values))
    print(stage, json.dumps(values), flush=True)


def prepare(args):
    cp = args.config.resolve(); c = read(cp)
    if (args.mode != 'approximate' or REPO / 'configs/approximate' not in cp.parents
            or c['reproduction_mode'] != 'approximate' or c['experiment_kind'] != 'graph_occlusion_research_v1'):
        raise ValueError('Explicit isolated APPROXIMATE configuration required')
    if c['metric_iou_threshold'] != .5 or c['runtime']['precision'] != 'float32':
        raise ValueError('Qualified bbox IoU0.5 / FP32 contract required')
    for key, expected in c['input_sha256'].items():
        if file_sha256(Path(c['paths'][key])) != expected: raise ValueError('Artifact drift: ' + key)
    source = Path(c['source_run']); old = read(source / 'protocol.json')
    if file_sha256(source / 'results.json') != c['source_results_sha256']:
        raise ValueError('Source result identity changed')
    if read(source / 'results.json')['status'] != 'SUCCESS' or read(source / 'independent_audit.json')['status'] != 'PASS':
        raise ValueError('Completed independently audited source required')
    for name in c['frozen_source_files']:
        if file_sha256(REPO / name) != old['source_code_sha256'][name]:
            raise ValueError('Frozen source implementation changed: ' + name)
    for name, expected in c['filter_source_sha256'].items():
        if file_sha256(Path(c['filter_source']) / name) != expected:
            raise ValueError('Qualified filter source changed: ' + name)
    if read(Path(c['filter_source']) / 'independent_audit.json')['status'] != 'PASS':
        raise ValueError('Filter audit failed')
    index = copy.deepcopy(old['index'])
    original_root = Path(c['paths']['source_word_root'])
    qualified_root = Path(c['paths']['word_root'])
    for phase, entries in index.items():
        if phase == 'test': continue
        for entry in entries:
            entry['image_path'] = str(qualified_root / Path(entry['image_path']).relative_to(original_root))
            entry['annotation_path'] = str(qualified_root / Path(entry['annotation_path']).relative_to(original_root))
    if [len(index[k]) for k in ('fit','calibration','evaluation','test')] != [1030,100,125,300]:
        raise ValueError('Qualified population changed')
    if args.smoke: index = {k:v[:c['smoke'][k]] for k,v in index.items()}
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    code = {str(p.relative_to(REPO)):file_sha256(p) for folder in ('models','localize','scripts','segmentation','recognition')
            for p in sorted((REPO / folder).rglob('*.py'))}
    identity = dict(config_sha256=file_sha256(cp), code_sha256=code, index=index, smoke=args.smoke)
    hardware = setup(c)
    if (out / 'protocol.json').exists():
        if read(out / 'protocol.json')['identity'] != identity: raise ValueError('Resume contract drift')
    else:
        write(out / 'protocol.json', dict(identity=identity, resolved_config=c, hardware=hardware,
            result_class='smoke-test result' if args.smoke else 'approximate reproduction; exploratory component experiment',
            metric=old['metric'], selection_uses_test=False, historical_test_exploration=True,
            known_train_test_duplicates=c['known_duplicate_test_ids'], seed_count=1,
            scope='No BLIP2/MPNet/DeepSolo routing; U-Net decoder replacement deferred'))
        write(out / 'resolved_config.json', c)
    return c, out, index, read(c['paths']['source_contract'])


def dataset_for(c, contract):
    manifest = load_segmentation_manifest(Path(c['paths']['train_manifest']), require_checksums=True)
    transform = _transform(contract['resolved_preprocessing'], 'letterbox', train=False)
    ds = WordKernelDataset(SegmentationDataset(manifest, Path(c['paths']['word_root']), transform=transform),
        distance_fraction=contract['resolved_config']['targets']['distance_fraction'])
    return ds, {s.sample_id:i for i,s in enumerate(manifest.samples)}


def filter_model(c):
    p = torch.load(Path(c['filter_source']) / 'selector/pointwise.pt', map_location='cpu', weights_only=True)
    dimensions = p['state_dict']['net.0.weight'].shape[1]
    model = CandidateScorer(dimensions, c['filter']['hidden'])
    model.load_state_dict(p['state_dict'], strict=True)
    return model.eval(), p['mean'], p['scale']


def budget_and_filter(c, records, ocr):
    """Pre-OCR ranking has no readings; cap unique crop pixels before inference."""
    pre = frozen_selector(c, 'pre'); model, mean, scale = filter_model(c); output = {}
    for sid, r in records.items():
        pool = r['arms']['mk']['pool']; rgb = load_rgb(r)
        for p in pool:
            x0,y0,x1,y1 = p['padded_bbox_original']
            p['crop_array_sha256'] = crop_hash(np.ascontiguousarray(rgb[y0:y1,x0:x1]))
        scores = score_record(r, 'mk', ocr, pre, False)
        eligible = preselect(r, 'mk', scores, c['budget'])
        candidates = [p for p in pool if p['id'] in eligible]
        ocr.image(rgb, candidates)
        if len({p['crop_array_sha256'] for p in candidates}) > c['budget']: raise ValueError('OCR budget exceeded')
        if candidates:
            x = torch.tensor([r['features']['mk'][p['id']] + ocr_features(ocr.rows[p['crop_array_sha256']]) for p in candidates])
            with torch.no_grad(): values = model((x - mean) / scale).sigmoid().tolist()
            output[sid] = compatible_selection(candidates, dict(zip([p['id'] for p in candidates],values)),
                c['filter']['threshold'], c['filter']['overlap'])
        else: output[sid] = []
    return output


def baseline_filter(c, records, phase, ocr):
    pred = budget_and_filter(c, records, ocr)
    expected = read(Path(c['filter_source']) / 'selector' / (phase + '_predictions.json'))['pointwise']
    for sid, pp in pred.items():
        if [p['id'] for p in pp] != [p['id'] for p in expected[sid]]:
            raise ValueError('60.70% reference filter failed exact replay: ' + sid)
    return pred


def graph_path(out, phase, sid):
    return out / 'graph_data' / phase / (sid.replace(':','__') + '.json')


def export_graph(c, out, index, contract):
    directory = out / 'graph_data'; directory.mkdir(exist_ok=False)
    model = source_model(c, contract); model.requires_grad_(False)
    ds, by = dataset_for(c, contract)
    transform = _transform(contract['resolved_preprocessing'], 'letterbox', train=False)
    pp = contract['resolved_config']['postprocessing']; g = c['graph']; count = 0
    torch.cuda.reset_peak_memory_stats(); fit_counts = dict(nodes=0, valid_nodes=0, positive_nodes=0, edges=0, positive_edges=0)
    for phase, entries in index.items():
        for entry in entries:
            sid = entry['sample_id']; rgb = load_rgb(entry)
            with torch.no_grad(): evidence = model.forward_evidence(_tensor(rgb, transform).cuda())
            prob = evidence['words'][0].sigmoid().cpu().numpy()
            deep = evidence['features'][0].cpu().numpy()
            source = read(path_for(Path(c['source_run']), phase, sid))
            if file_sha256(Path(source['maps_path'])) != source['maps_sha256']: raise ValueError('Source map bytes changed')
            with np.load(source['maps_path']) as saved:
                if not np.array_equal(prob, saved['probabilities']): raise ValueError('Frozen map replay failed: ' + sid)
            canvas = np.asarray(letterbox_canvas_mask((rgb.shape[1],rgb.shape[0]),native_size=prob.shape[1]), bool)
            instances = decode_instance_labels(prob, None, canvas, pp, None, 'cut')
            atoms, audit = fragment_labels(prob, instances, threshold=g['seed_threshold'],
                minimum_seed_area=g['minimum_seed_area'], maximum_pieces=g['maximum_pieces'], maximum_nodes=g['maximum_nodes'])
            boxes = []; values = []; rows = []
            for identity in range(1, int(atoms.max()) + 1):
                yy,xx = np.where(atoms == identity); box = [int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]
                original = remap_letterbox_native_box(box, rgb.shape[:2][::-1], native_size=atoms.shape[0])
                candidate = row(original, rgb.shape[:2], 'atom:' + str(identity), 'baseline')
                candidate['bbox_native'] = box; boxes.append(original); rows.append(candidate)
                vector = features(candidate, rgb, prob[0], text_threshold=pp['text_threshold'])
                # Fixed pretrained visual evidence, not labels or a learned second CNN.
                pooled = deep[:, yy, xx]
                values.append(vector + pooled.mean(1).tolist() + pooled.std(1).tolist())
            adjacency, edges = neighbor_graph(boxes, neighbors=g['neighbors'], maximum_distance=g['maximum_distance'])
            dimensions = len(features(row([0,0,1,1],rgb.shape[:2],'dummy','baseline'),rgb,prob[0],text_threshold=pp['text_threshold'])) + 2 * deep.shape[0]
            arrays = dict(features=np.asarray(values,np.float32).reshape(len(values),dimensions),
                adjacency=adjacency, edges=edges, atoms=atoms, probabilities=prob)
            if phase == 'fit':
                batch = ds[by[sid]]
                owners = batch['owners_native'].numpy(); valid = batch['validity'][0].numpy()
                if owners.shape != atoms.shape: raise ValueError('Owner/atom coordinate mismatch')
                assignments, eligible = graph_supervision(atoms, owners, valid, purity=g['owner_purity'])
                arrays.update(assignments=assignments, eligible=eligible)
                fit_counts['nodes'] += len(assignments); fit_counts['valid_nodes'] += int(eligible.sum())
                fit_counts['positive_nodes'] += int(((assignments > 0) & eligible).sum())
                edge_valid = eligible[edges[:,0]] & eligible[edges[:,1]]
                edge_positive = ((assignments[edges[:,0]] == assignments[edges[:,1]]) & (assignments[edges[:,0]] > 0))
                fit_counts['edges'] += int(edge_valid.sum()); fit_counts['positive_edges'] += int((edge_valid & edge_positive).sum())
            target = graph_path(out, phase, sid); target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target.with_suffix('.npz'), **arrays)
            write(target, dict(**entry, image_hw=list(rgb.shape[:2]), boxes=boxes, candidates=rows,
                arrays_sha256=file_sha256(target.with_suffix('.npz')), source_maps_sha256=source['maps_sha256'], audit=audit))
            count += 1
            if count % 50 == 0: announce(out, 'GRAPH_EXPORT', images=count, total=sum(map(len,index.values())))
    write(directory / 'memory.json', peaks('frozen U-Net text/kernel and 64-channel visual features',1,[1,3,512,512]))
    write(directory / 'complete.json', dict(status='SUCCESS', images=count, fit_supervision=fit_counts))
    del model, evidence; release()


def load_graph(out, phase, sid):
    target = graph_path(out, phase, sid); r = read(target)
    if file_sha256(target.with_suffix('.npz')) != r['arrays_sha256']: raise ValueError('Graph cache drift')
    with np.load(target.with_suffix('.npz')) as saved: arrays = {k:saved[k].copy() for k in saved.files}
    return r, arrays


def graph_records(model, mean, scale, c, out, entries, phase, policy, trace_dir):
    output = {}; model.eval(); trace_dir.mkdir(parents=True, exist_ok=True)
    pp = read(c['paths']['source_contract'])['resolved_config']['postprocessing']
    for entry in entries:
        sid = entry['sample_id']; r, z = load_graph(out, phase, sid)
        with torch.no_grad():
            node, edge = model((torch.from_numpy(z['features']).cuda()-mean)/scale,
                torch.from_numpy(z['adjacency']).cuda(),torch.from_numpy(z['edges']).cuda())
        ns, es = node.sigmoid().cpu().tolist(), edge.sigmoid().cpu().tolist()
        groups = graph_groups(r['boxes'], ns, z['edges'], es, node_threshold=policy['node'], edge_threshold=policy['edge'])
        rgb = load_rgb(r); pool = []
        for j, members in enumerate(groups):
            boxes = [r['boxes'][i] for i in members]
            box = [min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
            p = row(box,rgb.shape[:2], 'graph:' + str(j), 'merge' if len(members)>1 else 'baseline')
            p['atom_members'] = members; pool.append(p)
        record = {**entry, 'image_hw':r['image_hw'], 'arms':{'mk':{'pool':pool,'baseline':pool,'groups':[]}},
            'features':{'mk':{p['id']:features(p,rgb,z['probabilities'][0],text_threshold=pp['text_threshold']) for p in pool}}}
        write(trace_dir / (sid.replace(':','__') + '.json'),dict(record=record,node_probabilities=ns,
            edge_probabilities=es,edges=z['edges'].tolist(),policy=policy))
        output[sid] = record
    return output


def fit_graph(c, out, index, contract, smoke):
    directory = out / 'graph'; directory.mkdir(exist_ok=False); g = c['graph']
    data = []; vectors = []
    for entry in index['fit']:
        _, z = load_graph(out,'fit',entry['sample_id'])
        if len(z['features']): data.append(z); vectors.append(z['features'])
    x = torch.tensor(np.concatenate(vectors)); mean = x.mean(0).cuda(); scale = x.std(0).clamp_min(g['minimum_feature_scale']).cuda()
    packed = [{k:torch.from_numpy(v).cuda() for k,v in z.items() if k in ('features','adjacency','edges','assignments','eligible')} for z in data]
    for z in packed: z['features'] = (z['features']-mean)/scale
    source, old_ocr = source_records(c,index); ocr = CropOCR(c,directory / 'ocr.jsonl'); ocr.rows.update(old_ocr)
    initial = {phase:baseline_filter(c,source[phase],phase,ocr) for phase in ('calibration','evaluation','test')}
    all_results = {}; all_predictions = {}; selections = {}; capacities = {}
    for kind in ('self','message'):
        seed_everything(c['seed'],deterministic_algorithms=True)
        model = FragmentGraph(x.shape[1],g['hidden'],g['layers'],kind=='message').cuda()
        capacities[kind] = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(),lr=g['learning_rate'],weight_decay=g['weight_decay'])
        generator = torch.Generator().manual_seed(c['seed']); history = []; torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,(1 if smoke else g['epochs'])+1):
            model.train(); losses = []
            for j in torch.randperm(len(packed),generator=generator).tolist():
                z = packed[j]; assignments = z['assignments']; eligible = z['eligible']; edges = z['edges']
                optimizer.zero_grad(set_to_none=True); node, edge = model(z['features'],z['adjacency'],edges)
                same = (assignments[edges[:,0]] == assignments[edges[:,1]]) & (assignments[edges[:,0]] > 0)
                valid = eligible[edges[:,0]] & eligible[edges[:,1]]
                loss = balanced_binary_loss(node,(assignments>0).float(),eligible) + g['edge_loss_weight'] * balanced_binary_loss(edge,same.float(),valid)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite graph loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),g['gradient_clip'],error_if_nonfinite=True); optimizer.step()
                losses.append(float(loss.detach()))
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),updates=len(losses)))
            write(directory / (kind + '_history.json'),history)
            announce(out,'GRAPH_TRAIN',arm=kind,**history[-1])
        torch.save(dict(state_dict=model.state_dict(),mean=mean.cpu(),scale=scale.cpu(),parameters=capacities[kind]),directory / (kind + '.pt'))
        grid = []
        for policy in g['policies']:
            name = f"node{policy['node']}_edge{policy['edge']}"
            records = graph_records(model,mean,scale,c,out,index['calibration'],'calibration',policy,directory/kind/'calibration'/name)
            pred = budget_and_filter(c,records,ocr); result = report(records,pred,ocr.rows)
            grid.append(dict(policy=policy,aggregate=result['aggregate']))
            write(directory/kind/'calibration'/(name+'_predictions.json'),pred)
        best = max(range(len(grid)),key=lambda j:(grid[j]['aggregate']['f1'],-j)); selections[kind] = grid[best]['policy']
        write(directory / (kind + '_calibration.json'),grid)
        write(directory / (kind + '_selection_frozen.json'),dict(policy=selections[kind],epoch=len(history),checkpoint_selection='predeclared final epoch; policy uses calibration word F1'))
        for phase in ('evaluation','test'):
            records = graph_records(model,mean,scale,c,out,index[phase],phase,selections[kind],directory/kind/phase)
            pred = budget_and_filter(c,records,ocr)
            all_predictions[phase+':'+kind] = pred; all_results[phase+':'+kind] = report(records,pred,ocr.rows)
        write(directory / (kind + '_memory.json'),peaks('fragment graph + AdamW; frozen PARSeq during policy calibration',1,['variable_nodes',x.shape[1]]))
        del model, optimizer; ocr.release(); release()
    if capacities['self'] != capacities['message']: raise ValueError('Unmatched graph control capacity')
    for phase in ('evaluation','test'):
        predictions = {kind:all_predictions[phase+':'+kind] for kind in ('self','message')}
        predictions['baseline_filter'] = initial[phase]
        results = {kind:all_results[phase+':'+kind] for kind in ('self','message')}
        results['baseline_filter'] = report(source[phase],initial[phase],ocr.rows)
        write(directory / (phase+'_predictions.json'),predictions); write(directory / (phase+'_results.json'),results)
    write(directory / 'complete.json',dict(status='SUCCESS',selections=selections,parameters=capacities))


def recognized_segmentation(model,c,contract,entries,directory,ocr):
    records = export_model(model,c,contract,entries,directory / 'maps')
    predictions = budget_and_filter(c,records,ocr)
    write(directory / 'records.json',records); write(directory / 'predictions.json',predictions)
    return records, predictions, report(records,predictions,ocr.rows)


def fit_segmentation(c, out, index, contract, smoke):
    directory = out / 'segmentation'; directory.mkdir(exist_ok=False)
    sc = c['segmentation']; ds, by = dataset_for(c,contract)
    batches = [ds[by[e['sample_id']]] for e in index['fit']]
    source, old_ocr = source_records(c,index); ocr = CropOCR(c,directory/'ocr.jsonl'); ocr.rows.update(old_ocr)
    initial = {phase:baseline_filter(c,source[phase],phase,ocr) for phase in ('calibration','evaluation','test')}
    initial_f1 = report(source['calibration'],initial['calibration'],ocr.rows)['aggregate']['f1']
    audits = {}; all_results = {}; all_predictions = {}; selections = {}
    for arm in ('occlusion','pixel','structure'):
        seed_everything(c['seed'],deterministic_algorithms=True)
        student = source_model(c,contract); teacher = copy.deepcopy(student).requires_grad_(False).eval()
        original = {k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()}
        optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=sc['learning_rate'],weight_decay=sc['weight_decay'])
        generator = torch.Generator().manual_seed(c['seed']); branch = directory/arm; branch.mkdir()
        best = dict(epoch=0,f1=initial_f1); history = []; torch.cuda.reset_peak_memory_stats(); tick=time.perf_counter()
        epochs = 1 if smoke else sc['epochs']
        for epoch in range(1,epochs+1):
            student.train(); order = torch.randperm(len(batches),generator=generator).tolist(); losses = []; auxiliary = []
            for start in range(0,len(order),sc['batch_size']):
                chosen = [batches[j] for j in order[start:start+sc['batch_size']]]
                keys = ('image','mask','kernel','validity','separation','owners_native')
                batch = {k:torch.stack([b[k] for b in chosen]).cuda() for k in keys}
                flips = torch.rand((len(chosen),),generator=generator)<.5
                for k in batch: batch[k][flips] = batch[k][flips].flip(-1)
                weak = batch['image']
                contrast = 1 + float(torch.rand((),generator=generator)*2-1)*sc['contrast']
                brightness = float(torch.rand((),generator=generator)*2-1)*sc['brightness']
                strong, covered = occlude_instances(weak*contrast+brightness,batch['owners_native'],generator,
                    probability=sc['occlusion_probability'],width_fraction=sc['occlusion_width_fraction'],maximum_words=sc['maximum_occluded_words'])
                optimizer.zero_grad(set_to_none=True); logits=student(strong)
                loss = supervised_loss(logits,batch,contract['resolved_config']['loss']); extra = logits.sum()*0
                if arm != 'occlusion':
                    with torch.no_grad(): target = teacher(weak).sigmoid()
                    truth = torch.cat((batch['mask'],batch['kernel']),1)
                    reliable = teacher_reliability(target,truth,batch['validity'],sc['teacher_confidence'])
                    extra = sc['pixel_weight'] * pixel_distillation(logits,target,reliable)
                    if arm == 'structure':
                        extra = extra + sc['structure_weight'] * instance_structure_distillation(logits,target,reliable,batch['owners_native'],batch['separation'])
                    loss = loss + extra
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite occlusion training')
                loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],sc['gradient_clip'],error_if_nonfinite=True)
                optimizer.step(); losses.append(float(loss.detach())); auxiliary.append(float(extra.detach()))
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),distillation=float(np.mean(auxiliary)),updates=len(losses),seconds=time.perf_counter()-tick))
            write(branch/'history.json',history); optimizer.zero_grad(set_to_none=True)
            torch.save(dict(state_dict=student.state_dict(),epoch=epoch),branch/'last.pt')
            announce(out,'OCCLUSION_TRAIN',arm=arm,**history[-1])
            if epoch % sc['calibration_every'] == 0 or epoch == epochs:
                student.eval()
                _, _, metric = recognized_segmentation(student,c,contract,index['calibration'],branch/f'epoch{epoch}_calibration',ocr)
                write(branch/f'epoch{epoch}_calibration/results.json',metric)
                if metric['aggregate']['f1'] > best['f1']:
                    best = dict(epoch=epoch,f1=metric['aggregate']['f1'])
                    torch.save(dict(state_dict=student.state_dict(),selection=best),branch/'best.pt')
                write(branch/'current_selection.json',best)
                announce(out,'OCCLUSION_CALIBRATION',arm=arm,epoch=epoch,calibration_f1=metric['aggregate']['f1'],best=best)
                ocr.release()
        selections[arm] = best; write(branch/'selection_frozen.json',best)
        changed = [k for k,v in student.state_dict().items() if not torch.equal(v.detach().cpu(),original[k])]
        teacher_changed = [k for k,v in teacher.state_dict().items() if not torch.equal(v.detach().cpu(),original[k])]
        encoder_changed = [k for k in changed if k.startswith('encoder.')]
        if teacher_changed or encoder_changed or not changed: raise ValueError('Frozen teacher/encoder or student update contract failed')
        audits[arm] = dict(teacher_changed=teacher_changed,encoder_changed=encoder_changed,student_changed_tensors=len(changed),teacher_frozen=True)
        write(branch/'memory.json',peaks('M/K student + frozen clean teacher + AdamW; frozen PARSeq during calibration',sc['batch_size'],[sc['batch_size'],3,512,512]))
        del optimizer,teacher,student,original; release()
        for endpoint, checkpoint in ((arm,branch/'best.pt' if best['epoch'] else None),(arm+'_final',branch/'last.pt')):
            model = source_model(c,contract)
            if checkpoint is not None: model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'],strict=True)
            for phase in ('evaluation','test'):
                if checkpoint is None: predictions=initial[phase]; result=report(source[phase],predictions,ocr.rows)
                else:
                    _,predictions,result = recognized_segmentation(model,c,contract,index[phase],directory/endpoint/phase,ocr)
                all_results[phase+':'+endpoint]=result; all_predictions[phase+':'+endpoint]=predictions
            del model; ocr.release(); release()
    for phase in ('evaluation','test'):
        results = {name:all_results[phase+':'+name] for arm in selections for name in (arm,arm+'_final')}
        predictions = {name:all_predictions[phase+':'+name] for arm in selections for name in (arm,arm+'_final')}
        write(directory/(phase+'_results.json'),results); write(directory/(phase+'_predictions.json'),predictions)
    write(directory/'weight_audit.json',dict(status='PASS',arms=audits))
    write(directory/'complete.json',dict(status='SUCCESS',selections=selections))


def final_report(c,out,index,contract,smoke):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    records, readings = source_records(c,index)
    for path in (out/'graph/ocr.jsonl',out/'segmentation/ocr.jsonl'):
        for line in path.open():
            r=json.loads(line); key=r['crop_array_sha256']
            if key in readings and readings[key]['raw_text'] != r['raw_text']: raise ValueError('Frozen OCR transcript drift')
            readings[key]=r
    controls = dict(graph_self='baseline_filter',graph_message='graph_self',seg_occlusion='baseline_filter',
        seg_pixel='seg_occlusion',seg_structure='seg_pixel',seg_occlusion_final='baseline_filter',
        seg_pixel_final='seg_occlusion_final',seg_structure_final='seg_pixel_final')
    audit_count=0; summary={}; md=['# Fragment graph and occlusion teacher/student results','',
        '**Smoke-test result.**' if smoke else '**Approximate reproduction; exploratory component experiments.**','',
        'Word F1 uses exact Unicode casefold (punctuation retained) AND unpadded bbox IoU>=0.5. All care GT and predictions count. Historical TEST is explored; one seed.','']
    rng=np.random.default_rng(c['seed'])
    for phase in ('evaluation','test'):
        gt, hashes=labels_for(records[phase]); graph=read(out/'graph'/(phase+'_results.json')); gp=read(out/'graph'/(phase+'_predictions.json'))
        results={'baseline_filter':graph['baseline_filter']}; predictions={'baseline_filter':gp['baseline_filter']}
        for name in ('self','message'): results['graph_'+name]=graph[name]; predictions['graph_'+name]=gp[name]
        results.update({'seg_'+k:v for k,v in read(out/'segmentation'/(phase+'_results.json')).items()})
        predictions.update({'seg_'+k:v for k,v in read(out/'segmentation'/(phase+'_predictions.json')).items()})
        comparisons={}
        md += [f'## {phase}', '', '| Arm | F1 % | Control | Delta points | Recovered / damaged |','| --- | ---: | --- | ---: | ---: |']
        for name,result in results.items():
            per={r['sample_id']:r for r in result['per_image']}
            if set(per)!=set(gt) or set(predictions[name])!=set(gt): raise ValueError('Evaluation population drift')
            for sid,pp in predictions[name].items():
                matrix=np.array([[iou(p['bbox'],w['bbox'])>=.5 and readings[p['crop_array_sha256']]['raw_text'].casefold()==w['text'].casefold() for w in gt[sid]] for p in pp],bool).reshape(len(pp),len(gt[sid]))
                correct=int((maximum_bipartite_matching(csr_matrix(matrix),perm_type='column')>=0).sum()) if matrix.size else 0
                if correct!=per[sid]['correct'] or len(pp)!=per[sid]['proposals'] or len(gt[sid])!=per[sid]['gt_words']:
                    raise ValueError('Independent matching audit failed: '+name+' '+sid)
                audit_count+=1
            a=result['aggregate']
            if name not in controls:
                md.append(f"| {name} | {100*a['f1']:.3f} | — | — | — |")
                continue
            control=controls[name]; reference=results[control]; before={r['sample_id']:r for r in reference['per_image']}
            paired=changes(result,reference); detail=[]; numbers=[]
            for r in result['per_image']:
                b=before[r['sample_id']]; denominator=r['proposals']+r['gt_words']; old_denominator=b['proposals']+b['gt_words']
                delta=200*r['correct']/denominator-200*b['correct']/old_denominator if denominator and old_denominator else 0.
                detail.append(dict(sample_id=r['sample_id'],delta_f1_points=delta,
                    recovered_gt_indices=sorted(set(r['correct_gt_indices'])-set(b['correct_gt_indices'])),
                    damaged_gt_indices=sorted(set(b['correct_gt_indices'])-set(r['correct_gt_indices']))))
                numbers.append([r['correct'],r['proposals'],r['gt_words'],b['correct'],b['proposals']])
            array=np.array(numbers,float); draws=rng.integers(0,len(array),(c['bootstrap_draws'],len(array))); sums=array[draws].sum(1)
            boot=200*sums[:,0]/(sums[:,1]+sums[:,2])-200*sums[:,3]/(sums[:,4]+sums[:,2])
            value=dict(control=control,delta_f1_points=100*(a['f1']-reference['aggregate']['f1']),**paired,
                conditional_paired_95_percentile_interval=np.quantile(boot,[.025,.975]).tolist(),per_image=detail)
            comparisons[name]=value
            md.append(f"| {name} | {100*a['f1']:.3f} | {control} | {value['delta_f1_points']:+.3f} | {paired['recovered']} / {paired['damaged']} |")
        write(out/(phase+'_results.json'),results);write(out/(phase+'_predictions.json'),predictions)
        write(out/(phase+'_comparisons.json'),comparisons);write(out/(phase+'_annotation_hashes.json'),hashes)
        summary[phase]={k:v['aggregate'] for k,v in results.items()}
    write(out/'independent_audit.json',dict(status='PASS',image_arm_checks=audit_count,algorithm='independent SciPy maximum bipartite matching'))
    (out/'comparison.md').write_text('\n'.join(md)+'\n')
    write(out/'results.json',dict(status='SUCCESS',claim_label='approximate reproduction',smoke=smoke,results=summary))
    write(out/'status.json',dict(status='SUCCESS',phase='REPORT_COMPLETE',utc=utc(),pid=os.getpid()))
    print('REPORT_COMPLETE',json.dumps(summary),flush=True)


def record_failure(out, exc):
    write(out/'failure.json',dict(status='FAILED',utc=utc(),error=repr(exc),traceback=traceback.format_exc(),
        oom=isinstance(exc,torch.cuda.OutOfMemoryError)))
    write(out/'status.json',dict(status='FAILED',utc=utc(),error=repr(exc),oom=isinstance(exc,torch.cuda.OutOfMemoryError)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--smoke',action='store_true')
    p.add_argument('--stage',required=True,choices=['graph_export','graph','segmentation','report'])
    args=p.parse_args()
    try:
        c,out,index,contract=prepare(args);announce(out,args.stage.upper())
        if args.stage=='graph_export':export_graph(c,out,index,contract)
        elif args.stage=='graph':fit_graph(c,out,index,contract,args.smoke)
        elif args.stage=='segmentation':fit_segmentation(c,out,index,contract,args.smoke)
        else:final_report(c,out,index,contract,args.smoke)
    except Exception as exc:
        record_failure(args.output.resolve(),exc);raise


if __name__=='__main__':main()
