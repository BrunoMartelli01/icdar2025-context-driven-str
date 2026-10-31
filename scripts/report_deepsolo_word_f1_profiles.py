"""Export the global TRAIN winner and native-only TRAIN winner as explicit profiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate_deepsolo_word_f1_fusion import counts, paired_bootstrap, predictions
from recognition.deepsolo_word_f1 import aggregate
from scripts.run_deepsolo_word_f1_study import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=['approximate'])
    parser.add_argument('--config-output-root', required=True, type=Path)
    args = parser.parse_args()
    audit = read(args.run/'independent_audit.json')
    if audit['status'] != 'PASS' or audit['result_sha256'] != sha(args.run/'results.json'):
        raise ValueError('Require independent completed evaluation audit')
    selection = read(args.run/'selection.json'); processed = read(args.run/'processed_predictions.json')
    result = read(args.run/'results.json')
    source_cfg = selection['config']
    native_grid = [r for r in selection['train_grid'] if r['policy']['resolution'] == 'native']
    policies = dict(global_word=selection['selected']['word'],
                    native_word=max(native_grid, key=lambda r:r['word']['f1'])['policy'])
    from deepsolo.candidates import resolve_runtime_candidate
    resolved = resolve_runtime_candidate(read(args.runtime), source_cfg['candidate'])
    if 'approximate' not in args.config_output_root.parts:
        raise ValueError('Deployment profiles require the explicit APPROXIMATE namespace')
    report = dict(claim_label='approximate reproduction', reproduction_mode='approximate',
        official_evaluation=False, selection_split='train', test_fitting=False, profiles={})
    for name, policy in policies.items():
        cfg = dict(source_cfg, selected_policy=policy, profile=name,
            selection_rule='highest TRAIN word F1 within '+('all declared policies' if name == 'global_word' else 'declared native-only policies'),
            upstream_config_sha256=resolved['config_sha256'], upstream_base_config_sha256=resolved['base_config_sha256'],
            evaluation_result_sha256=sha(args.run/'results.json'), selection_sha256=sha(args.run/'selection.json'),
            profile_source_code_sha256=sha(Path(__file__)))
        path = args.config_output_root/('deepsolo_word_f1_selected_'+name+'_v1.json')
        save(path, cfg)
        report['profiles'][name] = dict(policy=policy, config_path=str(path), config_sha256=sha(path),
                                      selection_rule=cfg['selection_rule'], splits={})
        for split, rows in processed['splits'].items():
            split_result = {}
            for metric in ('word', 'localized'):
                selected = [counts(predictions(r, policy), r['ground_truth'], metric, source_cfg['matching_iou']) for r in rows]
                # Baseline is native threshold0.4 from the hash-pinned source plan,
                # already independently replayed by the main completed report.
                baseline = result['baseline'][split][metric]
                summary = aggregate(selected)
                split_result[metric] = dict(metrics=summary, delta_f1=summary['f1']-baseline['f1'])
                if split == 'test':
                    originals = result['protocols'][metric][split]['per_image']
                    if [r['sample_id'] for r in originals] != [r['sample_id'] for r in rows]:
                        raise ValueError('Bootstrap population/order changed')
                    split_result[metric]['uncertainty'] = paired_bootstrap([r['baseline'] for r in originals], selected,
                        source_cfg['bootstrap_repetitions'], source_cfg['seed'])
            report['profiles'][name]['splits'][split] = split_result
    save(args.run/'deployment_profiles.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
