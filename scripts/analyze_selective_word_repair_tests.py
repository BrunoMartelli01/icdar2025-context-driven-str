"""Post-hoc diagnostics of immutable selective-repair results; no policy changes."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json

from recognition.selective_word_repair import exact_matching, recovery_label
from recognition.text_reconstruction import lexical_changes
from scripts.run_selective_word_repair_tests import comparison
from scripts.run_text_reconstruction_tests import annotations, read, sha, write


def analyze(root,out):
    result=read(root/'results.json');c=read(root/'resolved_configuration.json');data=read(root/'samples.json')
    if result['status']!='SUCCESS' or read(root/'independent_audit.json')['status']!='PASS':
        raise ValueError('Completed and independently audited input required')
    out.mkdir(parents=True,exist_ok=False)
    analysis=dict(claim_label='approximate reproduction; post-hoc failure analysis',
        source_result_sha256=sha(root/'results.json'),analysis_code_sha256=sha(Path(__file__)),
        model_or_policy_updates=False,new_inference=False,new_training=False,cohorts={})
    for phase in ('evaluation','test'):
        samples=data[phase];baseline=read(root/'boundaries'/f'{phase}_baseline_predictions.json')
        gaps=read(root/'boundaries'/f'{phase}_gap_head_raw_predictions.json')
        triage=read(root/'selectors'/f'{phase}_triage_defer_predictions.json')
        added=read(root/'selectors'/f'{phase}_additions_predictions.json')
        reread=read(root/'reread'/f'{phase}_reread_selected_predictions.json')
        variants=read(root/'reread'/(phase+'_variants.json'))
        edit_kinds=Counter();gap_examples={k:[] for k in ('gain','harm','neutral')}
        reread_kinds=Counter();reread_examples=[];lost_triage=[];pool_count=expanded_count=gt_count=0
        geometry_count=expanded_geometry=restored_by_additions=novel_by_additions=0
        for sample in samples:
            sid=sample['sample_id'];words=annotations(sample);gt_count+=len(words)
            base_by={p['id']:p for p in baseline[sid]}
            for p in gaps[sid]:
                old=base_by[p['id']]
                if p['text']==old['text']:continue
                after=[p['text'] if q['id']==p['id'] else q['text'] for q in baseline[sid]]
                effect=lexical_changes([q['text'] for q in baseline[sid]],after,words)
                kind='harm' if effect['damaged'] else 'gain' if effect['recovered'] else 'neutral'
                edit_kinds[kind]+=1
                if len(gap_examples[kind])<10:
                    gap_examples[kind].append(dict(sample_id=sid,before=old['text'],after=p['text'],
                        bbox=p['bbox'],local_single_edit_effect=effect))
            old_matches=set(exact_matching(sample['baseline_raw'],words,c['matching_iou'])['matched_gt_indices'])
            filtered_matches=set(exact_matching(sample['baseline_filter'],words,c['matching_iou'])['matched_gt_indices'])
            added_matches=set(exact_matching(added[sid],words,c['matching_iou'])['matched_gt_indices'])
            recovered_additions=added_matches-filtered_matches
            restored_by_additions+=len(recovered_additions & old_matches)
            novel_by_additions+=len(recovered_additions-old_matches)
            triage_matches=set(exact_matching(triage[sid],words,c['matching_iou'])['matched_gt_indices'])
            for index in old_matches-triage_matches:
                lost_triage.append(dict(sample_id=sid,word=words[index]['text'],bbox=words[index]['bbox']))
            output_ids={p['id'] for p in reread[sid]}
            extra=[]
            for variant in variants[sid]:
                extra.extend(variant['children'])
                if {p['id'] for p in variant['children']}<=output_ids:
                    reread_kinds[variant['kind']]+=1
                    old=next(p for p in triage[sid] if p['id']==variant['parent'])
                    replacement=[p for p in triage[sid] if p['id']!=variant['parent']]+variant['children']
                    effect=recovery_label(triage[sid],replacement,words,c['matching_iou'])
                    reread_examples.append(dict(sample_id=sid,kind=variant['kind'],before=old,
                        after=variant['children'],local_single_variant_effect=effect))
            pool_count+=exact_matching(sample['pool'],words,c['matching_iou'])['correct']
            expanded_count+=exact_matching(sample['pool']+extra,words,c['matching_iou'])['correct']
            all_words=[dict(w,text='*') for w in words]
            geometry_count+=exact_matching([dict(p,text='*') for p in sample['pool']],all_words,c['matching_iou'])['correct']
            expanded_geometry+=exact_matching([dict(p,text='*') for p in sample['pool']+extra],all_words,c['matching_iou'])['correct']
        costs=result['reread_costs']['per_image'][phase]
        totals={k:sum(r[k] for r in costs) for k in ('admitted_source_unique_crops','routed_parents',
            'generated_variants','requested_unique_reread_crops','additional_cold_start_crops')}
        raw=result['cohorts'][phase]['baseline_raw']['spatial']
        defer=result['cohorts'][phase]['triage_defer']['spatial']
        add=result['cohorts'][phase]['additions']['spatial']
        filtered=result['cohorts'][phase]['baseline_filter']['spatial']
        selective=result['cohorts'][phase]['reread_selected']['spatial']
        analysis['cohorts'][phase]=dict(
            triage=dict(false_positives_removed=raw['fp']-defer['fp'],
                fraction_false_positives_removed=(raw['fp']-defer['fp'])/raw['fp'],
                correct_reads_lost=raw['correct']-defer['correct'],
                retained_correct_fraction=defer['correct']/raw['correct'],lost_correct_words=lost_triage),
            additions=dict(additional_outputs=add['predictions']-filtered['predictions'],
                recovered=add['correct']-filtered['correct'],
                restored_original_unfiltered_successes=restored_by_additions,
                new_successes_relative_to_original_unfiltered=novel_by_additions,
                incremental_precision=(add['correct']-filtered['correct'])/(add['predictions']-filtered['predictions'])
                    if add['predictions']!=filtered['predictions'] else None,
                no_additional_ocr=True),
            boundary_edits=dict(local_edit_categories=dict(edit_kinds),examples=gap_examples,
                interpretation='Single-edit effects keep all other observations original; they need not sum to the joint image effects.'),
            reread=dict(selected_variant_kinds=dict(reread_kinds),selected_variants=len(reread_examples),
                examples=reread_examples,costs=totals,
                cold_start_ocr_multiplier=(totals['admitted_source_unique_crops']+totals['additional_cold_start_crops'])/totals['admitted_source_unique_crops'],
                additional_correct=selective['correct']-defer['correct']),
            candidate_coverage_diagnostic=dict(original_admitted_pool_maximum_matching=pool_count,
                admitted_plus_all_reread_variants_maximum_matching=expanded_count,care_gt_words=gt_count,
                original_pool_geometry_maximum_matching=geometry_count,
                expanded_pool_geometry_maximum_matching=expanded_geometry,
                original_perfect_precision_f1_upper_bound=2*pool_count/(pool_count+gt_count),
                with_reread_perfect_precision_f1_upper_bound=2*expanded_count/(expanded_count+gt_count),
                note='Loose GT-assisted maximum matching ignores family incompatibilities and one-variant-per-parent constraints. The F1 bounds apply to saved transcriptions, not every future recognizer on these boxes. Not a deployable selector or achieved F1.'))
        add_report=read(root/'selectors'/f'{phase}_additions_results.json')
        gnn_report=read(root/f'{phase}_previous_additions_gnn_results.json')
        analysis['cohorts'][phase]['posthoc_hgb_vs_previous_gnn']=comparison(add_report,gnn_report,'spatial',c)
    train=read(root/'boundaries/training.json');history=read(root/'boundaries/history.json')
    analysis['boundary_training']=dict(**train,positive_gap_fraction=train['positive_gaps']/train['gap_positions'],
        initial_loss=history[0]['loss'],final_loss=history[-1]['loss'],updates=sum(r['updates'] for r in history),
        caveat='Falling FIT loss does not establish correct word boundaries on held-out images. Frozen encoder/head architecture differs from the historical full decoder fine-tune.')
    write(out/'diagnostics.json',analysis)
    t=analysis['cohorts']['test'];baseline=result['cohorts']['test']
    m={k:v['spatial'] for k,v in baseline.items() if k!='text_boundary_population'}
    bm={k:v['lexical'] for k,v in baseline['text_boundary_population'].items()}
    selector_training=read(root/'selectors/training.json');reread_training=read(root/'reread/training.json')
    hard_cal=read(root/'selectors/calibration_triage_hard_results.json')['aggregate']['spatial']
    fps=result['failure_analysis']['test']['false_positive_types']
    interval=result['comparisons']['test']['reread_selected']['spatial']['conditional_paired95_points']
    add_interval=result['comparisons']['test']['additions']['spatial']['conditional_paired95_points']
    boundary_selection=read(root/'boundaries/selection_frozen.json')['threshold']
    lines=['# Detailed selective word-repair analysis','',
        '**Approximate reproduction / exploratory analysis. Models and policies are unchanged.**','',
        f"All {len(data['test'])} TEST images / {m['baseline_raw']['gt_words']} care words are included. The budget20 track requires exact casefold text and original bbox IoU>=0.5. The original-block boundary track is lexical only.",'',
        '## Main findings','',
        f"- Conditional HGB additions recover {t['additions']['recovered']} words with no lost accepted observations. They add {t['additions']['additional_outputs']} outputs, incremental precision {t['additions']['incremental_precision']:.2%}. Spatial F1: {m['baseline_filter']['f1']:.2%} to {m['additions']['f1']:.2%}; conditional 95% delta interval [{add_interval[0]:+.2f},{add_interval[1]:+.2f}] points.",
        f"- Of the recovered additions, {t['additions']['restored_original_unfiltered_successes']} restore original unfiltered successes and {t['additions']['new_successes_relative_to_original_unfiltered']} are new relative to the unfiltered pipeline. All recovery counts are relative to the stated filtered control.",
        f"- Deferred rejection removes {t['triage']['false_positives_removed']} false positives ({t['triage']['fraction_false_positives_removed']:.2%}), losing {t['triage']['correct_reads_lost']} correct reads and retaining {t['triage']['retained_correct_fraction']:.2%} of original successes. Its F1 is {m['triage_defer']['f1']:.2%}; the older, more aggressive filter gives {m['baseline_filter']['f1']:.2%}.",
        f"- Hard rejection reaches {m['triage_hard']['f1']:.2%} F1 but loses {m['triage_hard']['damaged']} correct words. Its {hard_cal['damaged']} calibration losses must be compared with the declared deferred-policy damage budget {c['triage']['damage_limit']}. It is a fixed control, not a promoted safe policy.",
        f"- Selective rereading recovers {m['reread_selected']['recovered']} words with {m['reread_selected']['damaged']} damaged words relative to deferred rejection, F1 {m['triage_defer']['f1']:.2%} to {m['reread_selected']['f1']:.2%}. Conditional delta interval [{interval[0]:+.2f},{interval[1]:+.2f}] points omits model-search/training variability. Confidence-only replacement gives {m['confidence_only']['f1']:.2%}.",
        f"- The raw gap head changes spaces only, recovers {bm['gap_head_raw']['recovered']} words and damages {bm['gap_head_raw']['damaged']}. Outputs change {bm['baseline']['predictions']} to {bm['gap_head_raw']['predictions']}; F1 changes {bm['baseline']['f1']:.2%} to {bm['gap_head_raw']['f1']:.2%}. Calibration threshold: {boundary_selection} (>1 denotes KEEP). The previous generative ByT5 remains a distinct existing control at {bm['byt5_previous']['f1']:.2%}.",'',
        '## Why the errors remain','',
        f"The annotation-overlap proxy has {selector_training['text']['positives']} positives among {selector_training['text']['examples']} FIT candidates, versus {selector_training['correct']['positives']} exact-read positives. A candidate can overlap real text while still being incomplete, merged, or misread. Such cases are not solved by a background detector alone.",
        f"Original spatial false positives: {fps['baseline_raw']}. Deferred rejection: {fps['triage_defer']}. No-IoU cases can contain partial real text; they must not all be called background.",
        f"Rereading utility labels have {reread_training['benefit']['positives']} recovery positives and {reread_training['harm']['positives']} damage positives among {reread_training['benefit']['examples']} FIT variants. Inspect conditional intervals and both correct/false counts; higher recognition confidence alone is not proof of beneficial replacement.",
        f"The copied-character gap head has {train['pairs']} pairs and {train['positive_gaps']} positive gaps among {train['gap_positions']} positions; {train['source_target_conflicts']} source strings admit conflicting targets. FIT loss changes {history[0]['loss']:.4f} to {history[-1]['loss']:.4f}. Read the full calibration curve, not the training loss, for the selected behavior. This experiment does not prove every boundary model fails, nor isolate freezing the encoder as a cause.",'',
        '## Cost and candidate coverage','',
        f"TEST rereading routes {t['reread']['costs']['routed_parents']} parents and evaluates {t['reread']['costs']['generated_variants']} valid alternatives. It requires {t['reread']['costs']['additional_cold_start_crops']} additional unique RGB crop reads on top of {t['reread']['costs']['admitted_source_unique_crops']}, a {t['reread']['cold_start_ocr_multiplier']:.3f}x cold-start OCR count. Actual cache misses are not the production cost.",
        f"The loose original admitted pool supports {t['candidate_coverage_diagnostic']['original_admitted_pool_maximum_matching']} exact spatial matches. Adding all generated reread variants supports {t['candidate_coverage_diagnostic']['admitted_plus_all_reread_variants_maximum_matching']}. Perfect filtering bounds are {t['candidate_coverage_diagnostic']['original_perfect_precision_f1_upper_bound']:.2%} and {t['candidate_coverage_diagnostic']['with_reread_perfect_precision_f1_upper_bound']:.2%}. These use GT and ignore compatibility. Bounds apply to saved readings, not every possible future recognizer on these boxes.",
        f"Geometry-only maximum matching is {t['candidate_coverage_diagnostic']['original_pool_geometry_maximum_matching']} before and {t['candidate_coverage_diagnostic']['expanded_pool_geometry_maximum_matching']} after including rereads. The gap between geometric and exact-read matching includes incomplete/contaminated crops; it does not isolate pure recognizer errors.",
        'GPU stages run sequentially on the actual recorded hardware in FP32, without AMP/TF32. Read results.json memory records for exact allocated/reserved peaks, shapes, batches and timings. These are measured stage workloads, not an 8GiB hardware qualification or end-to-end latency benchmark.','',
        '## Scientific interpretation and next step','',
        'The strongest recovery signal is locked-output admission, while deferred rejection offers a better word-retention tradeoff than aggressive filtering. Selective visual rereading adds a small real recovery signal. A copied-character restriction alone does not make language corrections safe.',
        'Do not sum gains across the distinct baselines, adopt hard rejection solely because of its higher F1, or present the highest historical TEST endpoint as independent validation. Further work should first improve word-candidate coverage and conditional selection, and validate on data unused for model/policy exploration. More gap-head epochs or a larger fragment GNN are not supported by this battery.','',
        '## Verification and execution issues','',
        '60 focused/regression tests pass. Final smoke3 and the full independent audit pass. The first smoke failed because the new reader expected a nested source-manifest schema; the actual flat schema was restored without changing any input hash. A unit fixture also expected both synthetic and real provenance for an identical deduplicated pair; real provenance is now retained and a distinct real-spurious-space example tests both paths. Original failures are preserved.',
        'There was no actual OOM, dependency installation, new model download, source-dataset mutation, or strict/default configuration change. The controlled OOM unit test verifies failure propagation. All endpoint counts, new crop pixels, copy invariants and routing budgets have an independent check.','',
        'Read diagnostics.json for lost-word lists, corrected/damaged boundary examples, selected reread variants, exact source hashes, and the paired HGB/GNN comparison.']
    (out/'analysis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in t.items() if k not in ('boundary_edits','reread')},indent=2))
    print(json.dumps(dict(reread_kinds=t['reread']['selected_variant_kinds'],reread_variants=t['reread']['selected_variants']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',required=True,type=Path)
    parser.add_argument('--analysis-dir',required=True,type=Path)
    args=parser.parse_args();analyze(args.output_root.resolve(),args.analysis_dir.resolve())
