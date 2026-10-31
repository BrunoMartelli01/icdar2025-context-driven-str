"""Select a finite TRAIN-only word-U-Net policy, infer and certify feasibility."""

import argparse
import itertools
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from PIL import Image
import torch

from recognition.budget_target_oracle import analyze_budget
from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.f1_quality_study import aggregate
from recognition.icdar_budget_router import lexical_counts
from recognition.word_unet_policy import (
    apply_word_policy, normalized_changes, select_word_policy,
)
from segmentation.icdar2015_improvements import make_transform
from segmentation.icdar13_word_refit import WordRefitUNet
from scripts.analyze_budget_target_router import certify
from scripts.evaluate_icdar13_dense_instances import dictionary
from scripts.icdar_budget_data import read, sha, key, annotation, predictions
from scripts.run_icdar13_dense_instances import runtime
from scripts.train_icdar13_word_refit import read_rows, recognizer
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def checked_config(path):
    if path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate namespace required')
    config = read(path)
    assert config['mode'] == 'approximate'
    for name, digest in config['pinned_inputs'].items():
        assert sha(name) == digest, name
    return config


def prepare(path):
    config = checked_config(path)
    training = Path(config['training_root'])
    plan = read(training/'resolved_configuration.json')
    assert config['arms'] == plan['config']['arms']
    root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    source = {}
    for folder in ('models', 'segmentation', 'scripts', 'recognition', 'localize',
                   'caption', 'scoring', 'pipeline', 'research'):
        for file in (REPO/folder).rglob('*.py'):
            relative = file.relative_to(REPO)
            dest = root/'code_snapshot'/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, dest)
            source[str(relative)] = sha(dest)
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    seal(root/'resolved_configuration.json', dict(mode='approximate',
        claim_label='approximate reproduction', config=config,
        config_sha256=sha(path), source_sha256=source,
        training_plan_sha256=sha(training/'resolved_configuration.json'),
        historical_test_exposure=True, test_used_for_selection=False,
        metric='Per-image lexical multiset micro-F1, Unicode casefold+isalnum, '
               'native256 care override, all images/care GT retained; no IoU matching',
        result_class=plan['result_class']))
    print('PREPARED', root, flush=True)


def lookup(plan):
    return {r['sample_id']: r for r in plan['train_records']+plan['test_records']}


def fit_dictionary(config, plan, root):
    return dictionary(config, dict(partitions=plan['partitions'],
                                   records=lookup(plan)), root)


def select(config, plan, root):
    training = Path(config['training_root'])
    audit = read(training/'independent_audit.json')
    assert audit['status'] == 'PASS'
    if (root/'policy/frozen.json').exists():
        raise ValueError('Selection already frozen; do not overwrite it')
    lexicon = fit_dictionary(config, plan, root)
    records = lookup(plan)
    points = []
    ocr = recognizer(plan)
    try:
        for arm in config['arms']:
            complete = read(training/arm/'complete.json')
            checkpoint = training/arm/'best.pt'
            assert complete['checkpoint_sha256'] == sha(checkpoint)
            maps = training/arm/'selection'/(
                'epoch_'+str(complete['selected_epoch']).zfill(3))/'maps'
            for threshold, padding in itertools.product(config['text_thresholds'],
                                                         config['paddings']):
                setting = dict(plan['config']['inference'], text_threshold=threshold,
                               padding=padding, minimum_transcript_probability=0.)
                raw_dir = root/'policy/raw'/arm/(str(threshold)+'_pad'+str(padding))
                raw = {}
                for sid in plan['partitions']['selection']:
                    record = records[sid]
                    mp = maps/(key(sid)+'.npz')
                    saved = raw_dir/(key(sid)+'.json')
                    if saved.exists():
                        data = read(saved)
                        assert data['maps_sha256'] == sha(mp)
                        assert data['image_sha256'] == record['image_sha256']
                    else:
                        with np.load(mp, allow_pickle=False) as z:
                            probability = z['probabilities']
                        _, rows, expansion = read_rows(probability, record, setting,
                                                       ocr, plan['config']['parseq_batch_size'])
                        data = dict(rows=rows, expansion=expansion,
                            maps_path=str(mp), maps_sha256=sha(mp),
                            image_sha256=record['image_sha256'],
                            checkpoint_sha256=sha(checkpoint),
                            ground_truth_used_for_prediction=False)
                        seal(saved, data)
                    raw[sid] = data['rows']
                for sequence, lexical in itertools.product(config['sequence_thresholds'],
                                                             config['lexical_policies']):
                    policy = dict(setting, minimum_transcript_probability=sequence,
                                  lexical=lexical)
                    counts, changes = [], 0
                    for sid, rows in raw.items():
                        selected, trace = apply_word_policy(rows, policy, lexicon)
                        gt, valid = annotation(records[sid])
                        counts.append(lexical_counts(predictions(selected, records[sid], valid), gt))
                        changes += normalized_changes(trace)
                    points.append(dict(arm=arm, split='selection', policy=policy,
                        counts=aggregate(counts), normalized_changes=changes,
                        raw_directory=str(raw_dir), selected_epoch=complete['selected_epoch'],
                        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint)))
            print('SELECTION', arm, 'best_f1', max(p['counts']['f1'] for p in points
                if p['arm'] == arm), 'DeepSolo0/'+str(len(plan['partitions']['selection'])), flush=True)
    finally:
        ocr.release()
    best = select_word_policy(points)
    seal(root/'policy/grid.json', dict(points=points, deepsolo_images=0,
        total_images=len(plan['partitions']['selection']), deepsolo_percentage=0.))
    seal(root/'policy/frozen.json', dict(status='FROZEN', choice=best,
        test_used_for_selection=False, selection_ids=plan['partitions']['selection'],
        fit_dictionary_label_ids=plan['partitions']['fit'],
        fit_vocabulary_sha256=sha(root/'fit_generic_vocabulary.txt'),
        fit_pairs_sha256=sha(root/'fit_generic_pairs.json'),
        config_sha256=sha(root/'resolved_configuration.json'),
        training_audit_sha256=sha(training/'independent_audit.json'),
        promoted=best['counts']['f1'] >= config['minimum_selection_f1']))
    print('FROZEN', best, flush=True)


def load_selected(plan, frozen):
    choice = frozen['choice']
    assert sha(choice['checkpoint']) == choice['checkpoint_sha256']
    state = torch.load(choice['checkpoint'], map_location='cpu', weights_only=False)
    assert state['epoch'] == choice['selected_epoch']
    c = plan['config']
    net = WordRefitUNet(partial_encoder=False, skip_floor=c['skip_floor'],
                        initial_gap_bias=c['initial_gap_bias'])
    net.load_state_dict(state['model_state_dict'], strict=True)
    return net.to(c['device']).eval()


def infer(config, plan, root):
    frozen = read(root/'policy/frozen.json')
    assert frozen['promoted'] and not frozen['test_used_for_selection']
    assert frozen['config_sha256'] == sha(root/'resolved_configuration.json')
    selected_policy = frozen['choice']['policy']
    net = load_selected(plan, frozen)
    train_lexicon = fit_dictionary(config, plan, root)
    test_lexicon = DeepSoloLexicon.from_files(config['test_vocabulary'],
        config['test_pairs'], distance_exclusive=1.5)
    transform = make_transform(plan['config'], plan['config']['image_hw'], augment=False)
    records = lookup(plan)
    ocr = recognizer(plan)
    totals = {}
    torch.cuda.reset_peak_memory_stats()
    try:
        for role in config['inference_roles']:
            counts = []
            for sid in plan['partitions'][role]:
                record = records[sid]
                target = root/'selected_expert'/role/'samples'/(key(sid)+'.json')
                mp = root/'selected_expert'/role/'maps'/(key(sid)+'.npz')
                if target.exists():
                    sample = read(target)
                    assert sample['policy_sha256'] == sha(root/'policy/frozen.json')
                    assert sample['image_sha256'] == sha(record['image_path'])
                    assert sample['maps_sha256'] == sha(mp)
                else:
                    assert sha(record['image_path']) == record['image_sha256']
                    # The complete neural/crop/lexical path reads RGB only.
                    with Image.open(record['image_path']) as image:
                        rgb = image.convert('RGB')
                        x, _ = transform(rgb, Image.new('L', rgb.size, 0), 0)
                    with torch.inference_mode():
                        evidence = net.forward_evidence(x[None].to(plan['config']['device']))
                        p = evidence['words'][0].sigmoid()
                        feature = evidence['features'][0]
                        context = torch.cat((feature.mean((-2, -1)),
                            feature.std((-2, -1), correction=0),
                            (feature*p[:1]).sum((-2, -1))/p[0].sum().clamp_min(1e-12))).cpu().numpy()
                        probability = p.cpu().numpy()
                    mp.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(mp, probabilities=probability, shared_context=context)
                    _, raw, expansion = read_rows(probability, record, selected_policy,
                                                   ocr, plan['config']['parseq_batch_size'])
                    rows, trace = apply_word_policy(raw, selected_policy,
                        test_lexicon if role == 'test' else train_lexicon)
                    # GT is accessed only after a complete, frozen prediction.
                    gt, valid = annotation(record)
                    sample = dict(sample_id=sid, rows=rows, raw_rows=raw, trace=trace,
                        expansion=expansion, counts=lexical_counts(predictions(rows, record, valid), gt),
                        image_sha256=record['image_sha256'], maps_sha256=sha(mp),
                        policy_sha256=sha(root/'policy/frozen.json'),
                        ground_truth_used_for_prediction=False)
                    seal(target, sample)
                counts.append(sample['counts'])
            totals[role] = dict(counts=aggregate(counts), deepsolo_images=0,
                total_images=len(counts), deepsolo_percentage=0.)
            seal(root/'selected_expert'/role/'metrics.json', totals[role])
            print('EXPERT', role, totals[role], flush=True)
    finally:
        ocr.release()
    seal(root/'selected_expert/results.json', dict(status='COMPLETE', mode='approximate',
        claim_label='approximate reproduction', result_class=plan['result_class'],
        frozen_policy_sha256=sha(root/'policy/frozen.json'), roles=totals,
        actual_new_deepsolo_recognizer_calls=0, logical_cost='U-Net/PARSeq only',
        hardware=dict(gpu=torch.cuda.get_device_name(0), precision='float32',
            model_set='One complete Attention-Gated word U-Net and genuine PARSeq',
            image_shape=[1, 3, *plan['config']['image_hw']],
            parseq_batch=plan['config']['parseq_batch_size'],
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(), out_of_memory=False)))
    del net
    torch.cuda.empty_cache()


def oracle(config, plan, root):
    assert read(root/'selected_expert/results.json')['status'] == 'COMPLETE'
    teacher = read(config['fixed_deepsolo_counts'])
    assert teacher['checkpoint_sha256'] == config['fixed_deepsolo_checkpoint_sha256']
    records = lookup(plan)
    branches, sources = {}, {}
    for sid in plan['partitions']['test']:
        sample_path = root/'selected_expert/test/samples'/(key(sid)+'.json')
        sample = read(sample_path)
        reference = teacher['samples'][sid]
        assert reference['image_sha256'] == sample['image_sha256'] == records[sid]['image_sha256']
        evidence = reference['evidence_path']
        assert sha(evidence) == reference['evidence_sha256']
        branches[sid] = dict(light=sample['counts'], deepsolo=reference['counts'])
        sources[str(sample_path)] = sha(sample_path)
        sources[evidence] = sha(evidence)
    if not plan['smoke']:
        assert len(branches) == 233
        assert sum(v['light']['gt_words'] for v in branches.values()) == 1095
    budget = min(config['deepsolo_budget'], len(branches))
    report = analyze_budget(branches, list(branches), budget, config['target_f1'])
    assert certify(branches, list(branches), report, budget, config['target_f1'])
    seal(root/'oracle/results.json', dict(status='PASS', mode='approximate',
        claim_label='approximate reproduction', privileged=True,
        scope='TEST GT upper bound only; not a trained or deployable routing result',
        branches=branches, oracle=report, source_sha256=sources,
        expert_results_sha256=sha(root/'selected_expert/results.json'),
        fixed_deepsolo_counts_sha256=sha(config['fixed_deepsolo_counts']),
        actual_new_neural_inferences=0, routing_cost_scope='Logical choices on cached expert outputs'))
    print('ORACLE', {name: report[name] for name in ('target_feasible_within_budget',
        'minimum_deepsolo_images')}, 'F1', report['exact_budget']['f1'],
        'DeepSolo'+str(budget)+'/'+str(len(branches)), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--stage', choices=['prepare', 'select', 'infer', 'oracle'], required=True)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.config)
        return
    root = args.run_root
    evaluation = read(root/'resolved_configuration.json')
    config = evaluation['config']
    for name, digest in config['pinned_inputs'].items():
        assert sha(name) == digest, name
    for name, digest in evaluation['source_sha256'].items():
        assert sha(root/'code_snapshot'/name) == digest, name
    training_plan = Path(config['training_root'])/'resolved_configuration.json'
    assert sha(training_plan) == evaluation['training_plan_sha256']
    plan = read(training_plan)
    if args.stage != 'oracle':
        runtime(plan['config'])
    try:
        {'select': select, 'infer': infer, 'oracle': oracle}[args.stage](config, plan, root)
    except BaseException:
        seal(root/('failure_'+args.stage+'_'+str(time.time_ns())+'.json'),
             dict(status='FAILED', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
