"""Replay and audit native specialized-router results without model inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from recognition.clip_router_prompt_ablation import (
    budget_accept, coverage_threshold, fit_reference, image_scores)
from recognition.clip_router_threshold_sweep import validate_population
from recognition.paper_image_router_evaluation import word_metrics
from scripts import run_clip_router_prompt_ablation as frozen
from scripts.run_clip_high_recall_admission import payload_digest, seal
from scripts.crop_clip_runtime import sha
from scripts.run_clip_specialized_routers import CONFIG, PATHS, REPO


def read(path, digest=None):
    path = Path(path)
    if digest is not None and sha(path) != digest:
        raise ValueError('Audit input changed: ' + str(path))
    return json.loads(path.read_text())


def audit_model(name, root, cfg, observed, geometric, words):
    latest = read(root/name/'latest.json')
    path = Path(latest['results_path'])
    result = read(path, latest['results_sha256'])
    if (result['status'] != 'SUCCESS' or result['result_class'] != 'approximate reproduction'
            or result['selection_uses_test_labels'] or result['table_5_claim']):
        raise ValueError('A completed approximate full-population result is required')
    plan = read(path.parent/'resolved_configuration.json', result['plan_sha256'])
    if plan['config'] != cfg or plan['limit'] is not None:
        raise ValueError('Audit protocol differs from experiment protocol')
    for relative, digest in plan['code_sha256'].items():
        if sha(path.parent/'code_snapshot'/relative) != digest:
            raise ValueError('Experiment code snapshot changed')
    for source, digest in plan['source_sha256'].items():
        read(source, digest)
    selection = read(path.parent/'train_selection.json', result['train_selection_sha256'])
    expected_ids = {s:[r['sample_id'] for r in observed[s]] for s in ('train','test')}
    if result['split_ids'] != expected_ids:
        raise ValueError('Full split population/order changed')
    partition = frozen.partition(expected_ids['train'],cfg['calibration_images'],cfg['execution']['seed'])
    if (selection['fit_ids'] != partition['calibration']
            or selection['selection_ids'] != partition['validation']
            or selection['selection_uses_test_labels']):
        raise ValueError('TRAIN-only fit/selection contract changed')
    features = {}
    for split, ids in expected_ids.items():
        for sid in ids:
            feature = read(path.parent/split/(sid.replace(':','__')+'.json'))
            if (feature['payload_sha256'] != payload_digest(feature)
                    or feature['plan_sha256'] != result['plan_sha256']
                    or feature['sample_id'] != sid or feature['ground_truth_in_scoring']
                    or not feature['routing_includes_ignored_words']
                    or feature['runtime']['out_of_memory']):
                raise ValueError('Native feature contract changed: ' + sid)
            features[sid] = feature
    reference = fit_reference(features, partition['calibration'])
    if reference != selection['reference']:
        raise ValueError('TRAIN calibration reference replay differs')
    scores = {sid:image_scores(feature, reference) for sid,feature in features.items()}
    replayed = 0
    for family, fitted in selection['families'].items():
        values = {sid:s[family] for sid,s in scores.items()}
        cut = coverage_threshold(values,partition['calibration'],cfg['coverage'])
        rows = [r for r in geometric['train'] if r['sample_id'] in partition['validation']]
        validation = frozen.evaluate(rows,values,cut,partition['validation'],sum(r['parseq']['gt_words'] for r in rows))
        if fitted != {'threshold':cut,'validation':validation}:
            raise ValueError('TRAIN family selection replay differs')
        for split, ids in expected_ids.items():
            keep = budget_accept(values,ids,cfg['comparison_budget'][split])
            policies = {'train_threshold':(values,cut),
                'same_budget_descriptive':({sid:float(sid in keep) for sid in ids},1.)}
            for data, key in ((words,'comparisons'),(geometric,'geometric_comparisons')):
                for policy, (v,t) in policies.items():
                    point = frozen.evaluate(data[split],v,t,ids,cfg['gt_words'][split])
                    if point != result[key][family][split][policy]:
                        raise ValueError('Metric replay differs: ' + repr((name,family,split,policy,key)))
                    replayed += 1
    winner = max(selection['families'],key=lambda f:(
        selection['families'][f]['validation']['precision_retained_parseq'] or -1,
        selection['families'][f]['validation']['words_retained_parseq'],f))
    if winner != result['winner_train'] or winner != selection['winner']:
        raise ValueError('TRAIN winner replay differs')
    peaks = {key:max((f['runtime']['memory'][key] for f in features.values()
                     if f['runtime']['memory'][key] is not None),default=None)
             for key in result['memory']}
    if peaks != result['memory']:
        raise ValueError('Native memory receipt differs')
    return result, scores, {'features_verified':len(features),'metric_points_replayed':replayed,
        'results_path':str(path),'results_sha256':sha(path),'plan_sha256':result['plan_sha256'],
        'source_snapshot_files':len(plan['code_sha256']),'memory':peaks}


def table_row(model, family, split, policy, point):
    combined = point['combined_model'] or {}
    return {'model':model,'family':family,'split':split,'policy':policy,
        'threshold':point['threshold'],'accepted_images':point['parseq_images'],
        'parseq_words':point['words_retained_parseq'],
        'parseq_correct_words':point['correct_words_retained_parseq'],
        'parseq_word_precision':point['precision_retained_parseq'],
        'safe_images':point['safe_images'],'complete_images':point['complete_images'],
        'missing_fallback_images':len(point['missing_deepsolo_sample_ids']),
        'total_predicted_words':combined.get('detected_words'),
        'combined_precision':combined.get('accuracy_on_detected_words'),
        'combined_tn0':combined.get('accuracy_tn0'),'combined_f1':combined.get('word_f1')}


def exact_budget_curve(rows, values, ids, gt):
    """Exhaustive descriptive curve; ties use only deterministic sample IDs."""
    ranked = sorted((sid for sid in ids if values[sid] is not None),
                    key=lambda sid:(-values[sid],sid))
    points = []
    for count in range(len(ranked)+1):
        keep = set(ranked[:count])
        point = frozen.evaluate(rows,{sid:float(sid in keep) for sid in ids},1.,ids,gt)
        cut = values[ranked[count-1]] if count else None
        point.update(rank_cut_score=cut,
            inclusive_scalar_accepts=sum(values[sid]>=cut for sid in ranked) if count else 0,
            threshold_is_binary_budget_encoding=True)
        points.append(point)
    return points


def validate_completed_branches(original, completed):
    for split in ('train','test'):
        before, after = original['splits'][split], completed['splits'][split]
        ids = [r['sample_id'] for r in before]
        validate_population(after,ids,sum(r['parseq']['gt_words'] for r in before))
        if [r['sample_id'] for r in after] != ids:
            raise ValueError('Supplemented split ordering changed')
        for old,new in zip(before,after):
            if ({k:v for k,v in old.items() if k != 'deepsolo'}
                    != {k:v for k,v in new.items() if k != 'deepsolo'}):
                raise ValueError('Supplement modified frozen non-DeepSolo data')
            if (old['deepsolo'] is not None and new['deepsolo'] != old['deepsolo']):
                raise ValueError('Supplement replaced an existing DeepSolo output')
            if new['deepsolo'] is None:
                raise ValueError('Supplement leaves missing DeepSolo outputs')


def load_train_completion(root, word):
    latest = read(root/'latest.json')
    path = Path(latest['completion_path'])
    receipt = read(path,latest['completion_sha256'])
    if receipt['status'] != 'SUCCESS':
        raise ValueError('Complete TRAIN supplement required')
    plan = read(path.parent/'resolved_configuration.json',receipt['plan_sha256'])
    if sha(Path(plan['source_word_branches'])) != receipt['source_word_branch_sha256']:
        raise ValueError('Supplement source word branches changed')
    for name,digest in plan['code_sha256'].items():
        if sha(path.parent/'code_snapshot'/name) != digest:
            raise ValueError('Supplement source snapshot changed')
    completed = read(path.parent/'completed_word_branches.json',receipt['word_branch_sha256'])
    for source,digest in completed['input_sha256'].items():
        read(source,digest)
    validate_completed_branches(word,completed)
    needed = {r['sample_id'] for r in word['splits']['train'] if r['deepsolo'] is None}
    if set(receipt['per_image']) != needed or receipt['new_train_inferences'] != len(needed):
        raise ValueError('Unexpected supplemental inference population')
    by_id = {r['sample_id']:r for r in completed['splits']['train']}
    for sid,record in receipt['per_image'].items():
        raw = read(record['response_path'],record['response_sha256'])
        if raw['payload_sha256'] != payload_digest(raw) or not raw['response']['success']:
            raise ValueError('Supplemental raw response failed its contract')
        metric = word_metrics(record['predictions'],by_id[sid]['ground_truth'])
        if metric != record['word'] or metric != by_id[sid]['deepsolo']:
            raise ValueError('Supplemental word counts do not replay')
    return completed['splits'],{'completion_path':str(path),'sha256':sha(path),
        'new_train_inferences':len(needed),'word_branch_sha256':receipt['word_branch_sha256'],
        'gates_refitted':False,'test_branches_unchanged':completed['splits']['test']==word['splits']['test']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['approximate'])
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--paths',type=Path,default=PATHS)
    parser.add_argument('--models',nargs='+',default=['longclip_b','longclip_l','llm2clip_l14_336'])
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--previous-audit',type=Path)
    parser.add_argument('--train-fallback-supplement',type=Path)
    parser.add_argument('--pretest-selection',type=Path)
    parser.add_argument('--paper-baseline',type=Path)
    args = parser.parse_args()
    destination = args.output_root.resolve()
    if destination.is_relative_to(REPO):
        parser.error('Audit outputs must be outside the repository')
    # The frozen loader uses output_root for inference; this audit only reads its
    # configured inference location and writes to its separate destination.
    args.output_root = None
    cfg, paths, observed, geometric, source_hashes = frozen.load_protocol(args)
    word = read(paths['word_branches'],cfg['word_branch_sha256'])
    for source,digest in word['input_sha256'].items():
        read(source,digest)
    words = word['splits']
    completed_words, completion_receipt = (load_train_completion(args.train_fallback_supplement,word)
        if args.train_fallback_supplement else (words,None))
    results, receipts, curves, budgets, table, bootstrap = {}, {}, {}, {}, [], {}
    baseline = None
    native_control = None
    for name in args.models:
        result, scores, receipt = audit_model(name,paths['output_root'],cfg,observed,geometric,words)
        results[name], receipts[name] = result, receipt
        if native_control is not None and result['comparisons']['native_only'] != native_control:
            raise ValueError('PARSeq-only confidence control differs across encoders')
        native_control = result['comparisons']['native_only']
        if baseline is None:
            baseline = result['original_router']
        elif baseline != result['original_router']:
            raise ValueError('Original router baseline differs between encoders')
        winner = result['winner_train']
        curves[name] = {}
        budgets[name] = {}
        for split in ('train','test'):
            ids = result['split_ids'][split]
            values = {sid:s[winner] for sid,s in scores.items()}
            cut = result['comparisons'][winner][split]['train_threshold']['threshold']
            thresholds = sorted(set(cfg['report_thresholds']) | {cut})
            curves[name][split] = [frozen.evaluate(completed_words[split],values,t,ids,cfg['gt_words'][split]) for t in thresholds]
            budgets[name][split] = exact_budget_curve(completed_words[split],values,ids,cfg['gt_words'][split])
            table.extend(table_row(name,winner,split,'threshold_curve',p) for p in curves[name][split])
            for family in sorted({winner,'native_only','crop_baseline','scene_plain','scene_all_en','scene_reads_en'}):
                for policy,point in result['comparisons'][family][split].items():
                    table.append(table_row(name,family,split,policy,point))
        if completion_receipt:
            result['completed_train_comparisons'] = {}
            for family in result['comparisons']:
                values = {sid:s[family] for sid,s in scores.items()}
                ids = result['split_ids']['train']
                keep = budget_accept(values,ids,cfg['comparison_budget']['train'])
                cut = result['comparisons'][family]['train']['train_threshold']['threshold']
                policies = {'train_threshold':(values,cut),
                    'same_budget_descriptive':({sid:float(sid in keep) for sid in ids},1.)}
                points = {policy:frozen.evaluate(completed_words['train'],v,t,ids,cfg['gt_words']['train'])
                          for policy,(v,t) in policies.items()}
                result['completed_train_comparisons'][family] = points
                if family in {winner,'native_only','crop_baseline','scene_plain','scene_all_en','scene_reads_en'}:
                    table.extend(table_row(name,family,'train',p+'_complete_train',point)
                                 for p,point in points.items())
        # Conditional paired image bootstrap for the already frozen gates.
        # It does not re-fit ranks or use TEST labels to choose a policy.
        keep = set(result['comparisons'][winner]['test']['same_budget_descriptive']['accepted_sample_ids'])
        old_keep = set(baseline['test']['accepted_sample_ids'])
        arrays = [np.asarray([[r['parseq' if r['sample_id'] in gate else 'deepsolo'][key]
                  for key in ('tp','fp','fn')] for r in words['test']],dtype=np.int64)
                  for gate in (keep,old_keep)]
        expected = [result['comparisons'][winner]['test']['same_budget_descriptive']['combined_model'],
                    baseline['test']['combined_model']]
        for array, metric in zip(arrays,expected):
            if array.sum(axis=0).tolist() != [metric[k] for k in ('tp','fp','fn')]:
                raise ValueError('Paired bootstrap counts differ from audited model metrics')
        rng = np.random.default_rng(cfg['execution']['seed'])
        differences = []
        for _ in range(5):
            draws = rng.integers(0,len(words['test']),(1000,len(words['test'])))
            counts = [a[draws].sum(axis=1) for a in arrays]
            differences.extend(100*(counts[0][:,0]/counts[0].sum(1)-counts[1][:,0]/counts[1].sum(1)))
        bootstrap[name] = {'samples':5000,'seed':cfg['execution']['seed'],
            'unit':'paired TEST image, frozen gates','test_is_exploratory':True,
            'tn0_delta_vs_original_pp':100*(result['comparisons'][winner]['test']['same_budget_descriptive']['combined_model']['accuracy_tn0']-baseline['test']['combined_model']['accuracy_tn0']),
            'percentile_95_interval_pp':np.quantile(differences,[.025,.975]).tolist()}
    for split, point in baseline.items():
        table.append(table_row('original_vit_b16','original_frozen_gate',split,'original_threshold',point))
    budgets['original_vit_b16'] = {}
    for split in ('train','test'):
        ids = [r['sample_id'] for r in words[split]]
        values = {r['sample_id']:r['score'] for r in geometric[split]}
        budgets['original_vit_b16'][split] = exact_budget_curve(completed_words[split],values,ids,cfg['gt_words'][split])
    controls = {}
    for name, engine in [('deepsolo_only','deepsolo'),('parseq_only','parseq')]:
        controls[name] = {}
        for split in ('train','test'):
            ids = [r['sample_id'] for r in words[split]]
            values = dict.fromkeys(ids,1. if engine=='parseq' else 0.)
            point = frozen.evaluate(completed_words[split],values,1.,ids,cfg['gt_words'][split])
            controls[name][split] = point
            table.append(table_row(name,name,split,'all_images',point))
    report = {'status':'PASS','claim_label':'approximate reproduction',
        'reproduction_mode':'approximate','result_class':'approximate reproduction','table_5_claim':False,
        'selection_uses_test_labels':False,'same_budget_test_is_descriptive':True,
        'metric_protocol':next(iter(results.values()))['metric_protocol'],
        'input_word_branch_sha256':cfg['word_branch_sha256'],
        'train_fallback_supplement':completion_receipt,
        'audit_code_sha256':sha(Path(__file__)),'audit_loader_sources':source_hashes,
        'models':results,'original_router':baseline,'controls':controls,
        'verification':receipts,'paired_bootstrap':bootstrap,
        'parseq_native_control_equal_across_encoders':True,
        'threshold_curves':curves,'exact_budget_curves':budgets,'table':table}
    if args.previous_audit:
        previous_path = args.previous_audit/'comparison.json'
        previous = read(previous_path)
        prior_receipt = read(args.previous_audit/'completion_receipt.json')
        if (sha(previous_path) != prior_receipt['artifact_sha256']['comparison.json']
                or previous['word_branch_sha256'] != cfg['word_branch_sha256']):
            raise ValueError('Prior comparison changed or uses a different word metric')
        report['previous_comparison'] = {'path':str(previous_path),'sha256':sha(previous_path),
            'models':previous['models']}
        if completion_receipt:
            report['previous_completed_train'] = {}
            ids = [r['sample_id'] for r in completed_words['train']]
            for name,model in previous['models'].items():
                family = model['winner_train']
                report['previous_completed_train'][name] = {}
                for policy,old in model['comparisons'][family]['train'].items():
                    keep = set(old['accepted_sample_ids'])
                    point = frozen.evaluate(completed_words['train'],
                        {sid:float(sid in keep) for sid in ids},1.,ids,cfg['gt_words']['train'])
                    for key in ('words_retained_parseq','correct_words_retained_parseq','precision_retained_parseq'):
                        if point[key] != old[key]:
                            raise ValueError('Supplement changed the retained words of a prior gate')
                    report['previous_completed_train'][name][policy] = point
    if args.pretest_selection:
        receipt = read(args.pretest_selection)
        matching = [name for name,r in results.items() if r['plan_sha256']==receipt['plan_sha256']]
        if (len(matching)!=1 or receipt['test_labels_used']
                or receipt['test_feature_count_at_capture']>=receipt['expected_test_images']):
            raise ValueError('Pre-TEST selection receipt is incompatible')
        name = matching[0]
        if results[name]['winner_train'] != receipt['winner']:
            raise ValueError('Final winner differs from the pre-TEST TRAIN winner')
        for source,digest in receipt['train_feature_sha256'].items():
            read(source,digest)
        report['pretest_selection_verified'] = {'model':name,'receipt_path':str(args.pretest_selection),
            'sha256':sha(args.pretest_selection),'winner':receipt['winner'],
            'test_features_at_capture':receipt['test_feature_count_at_capture']}
    if args.paper_baseline:
        audit = read(args.paper_baseline/'completion_audit.json')
        paper = read(args.paper_baseline/'results.json',audit['result_sha256'])
        read(args.paper_baseline/'resolved_configuration.json',paper['plan_sha256'])
        if (audit['status']!='PASS' or paper['selection_uses_test_labels']
                or sha(args.paper_baseline/'word_branch_metrics.json')!=cfg['word_branch_sha256']):
            raise ValueError('Paper-style baseline is not compatible with the frozen word protocol')
        paper_points = {}
        for profile,gates in paper['profiles'].items():
            paper_points[profile] = {}
            for gate,record in gates.items():
                paper_points[profile][gate] = {}
                for split in ('train','test'):
                    ids = [r['sample_id'] for r in completed_words[split]]
                    points = {}
                    for policy,old in record['splits'][split]['points'].items():
                        keep = set(old['accepted_sample_ids'])
                        sources = old['accepted_sample_sources']
                        rows = [{**r,'parseq':r['t1'] if sources.get(r['sample_id'])=='T1' else r['parseq']}
                                for r in completed_words[split]]
                        point = frozen.evaluate(rows,{sid:float(sid in keep) for sid in ids},1.,ids,cfg['gt_words'][split])
                        for key in ('words_retained_parseq','correct_words_retained_parseq','precision_retained_parseq'):
                            if point[key]!=old[key]:
                                raise ValueError('Frozen paper-style accepted predictions changed')
                        if split=='test':
                            for key in ('tp','fp','fn','gt_words','detected_words'):
                                if point['combined_model'][key]!=old['combined_model'][key]:
                                    raise ValueError('Frozen paper-style TEST replay changed')
                        point.update(threshold=old['threshold'],
                            gate_replayed_from_frozen_accepted_ids=True,
                            t1_images=old['t1_images'],t3_images=old['t3_images'])
                        points[policy] = point
                        if profile==paper['primary_caption_profile'] and gate==paper['primary_gate']:
                            table.append(table_row('paper_style_MPNet',gate,split,policy,point))
                    paper_points[profile][gate][split] = points
        report['paper_style_baseline'] = {'source_path':str(args.paper_baseline),
            'results_sha256':audit['result_sha256'],'primary_profile':paper['primary_caption_profile'],
            'primary_gate':paper['primary_gate'],'refitted':False,'points':paper_points}
    destination.mkdir(parents=True,exist_ok=True)
    seal(destination/'comparison.json',report)
    with (destination/'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps({'status':'PASS','models':args.models,'output':str(destination),
        'feature_payloads_verified':sum(r['features_verified'] for r in receipts.values()),
        'metric_points_replayed':sum(r['metric_points_replayed'] for r in receipts.values())}),flush=True)


if __name__ == '__main__':
    main()
