"""Create a GT-free DeepSolo export plan for an already frozen exact image quota."""

import argparse
from pathlib import Path

from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def selected_records(records, decision, total, fallback):
    by_id = {r['sample_id']: r for r in records}
    selected, accepted = decision['fallback_ids'], decision['accepted_ids']
    if (len(records) != total or len(by_id) != total or len(selected) != fallback or
            len(set(selected)) != fallback or len(accepted) != total - fallback or
            len(set(accepted)) != len(accepted) or set(selected) & set(accepted) or
            set(selected) | set(accepted) != set(by_id) or decision['ground_truth_used'] or
            decision['fallback_outputs_loaded']):
        raise ValueError('Expected a complete GT-free frozen exact-budget route')
    return [{k: by_id[s][k] for k in ('sample_id', 'image_path', 'image_sha256')}
            for s in sorted(selected)]


def run(path):
    if path.resolve().parent != REPO / 'configs/approximate':
        raise ValueError('Explicit approximate DeepSolo export configuration required')
    config = read(path)
    if config['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    for file, digest in config['pinned_inputs'].items():
        if sha(file) != digest:
            raise ValueError('Changed frozen selected-export input: ' + file)
    original = read(config['source_plan']); decision = read(config['decision_path'])
    records = selected_records(original['population']['test'][config['dataset']], decision,
        config['total_images'], config['deepsolo_images'])
    original_config = original['config']; paths = original_config['paths']
    export = dict(mode='approximate', claim_label='approximate reproduction',
        config=dict(mode='approximate', minimum_free_gpu_bytes=original_config['minimum_free_gpu_bytes'],
            paths={k: paths[k] for k in ('deepsolo_profile', 'deepsolo_runtime', 'deepsolo_source')}),
        population=dict(test={config['dataset']: records}),
        selected_before_inference=True, decision_path=config['decision_path'],
        decision_sha256=sha(config['decision_path']), source_plan_sha256=sha(config['source_plan']),
        config_sha256=sha(path), ground_truth_paths_included=False,
        deepsolo_images=len(records), total_images=config['total_images'],
        deepsolo_percentage=100 * len(records) / config['total_images'],
        historical_test_exposure=True, scientific_choices=config['scientific_choices'])
    out = Path(config['output_plan'])
    if out.exists():
        raise ValueError('Frozen selected-export plan already exists')
    seal(out, export)
    print('FROZEN GT-FREE DEEPSOLO EXPORT', len(records), config['total_images'],
          100 * len(records) / config['total_images'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    run(p.parse_args().config)
