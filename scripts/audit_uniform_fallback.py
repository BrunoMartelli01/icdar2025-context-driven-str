"""Verify prediction-only fallback processing against every qualified TEST cache."""

import argparse
import json
import hashlib
from pathlib import Path
import time

from recognition.uniform_fallback import UniformFallbackPostprocessor
from scripts.evaluate_uniform_unet_fusion import fallback_rows
from scripts.run_uniform_unet_training import read, write
from segmentation.uniform_unet_fusion import file_hash


def audit(config_path):
    c = read(config_path)
    if c['mode'] != 'approximate' or config_path.resolve().parent.name != 'approximate':
        raise ValueError('Explicit approximate fallback configuration required')
    for path, digest in c['pinned_inputs'].items():
        if file_hash(path) != digest:
            raise ValueError('Changed fallback source: '+path)
    e = read(c['evaluation_config']); index = read(c['raw_index']); checks = {}
    for ds, profile in c['profiles'].items():
        processor = UniformFallbackPostprocessor(profile)
        plan = read(Path(c['training_root'])/ds/'resolved_configuration.json')
        ids = plan['partitions']['test']; lookup = {r['sample_id']: r for r in plan['test_records']}
        assert set(index[ds]) == set(ids) and len(ids) == e['expected_images'][ds]
        spec = e['fallback_test_sources'][ds]
        archive = read(spec['path']) if spec['kind'] != 'icdar13_files' else None
        rows_checked = 0
        for sid in ids:
            entry = index[ds][sid]
            assert file_hash(entry['path']) == entry['sha256']
            raw = read(entry['path'])
            content = {k: v for k, v in raw.items() if k != 'payload_sha256'}
            assert raw['payload_sha256'] == hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
            assert raw['contract']['sample_id'] == sid and raw['contract']['image_sha256'] == lookup[sid]['image_sha256']
            # This deployed function receives only a raw neural response, RGB
            # dimensions/orientation and a bound checkpoint hash. No GT/ignore.
            result = processor.apply(raw['response'], lookup[sid]['image_wh'], lookup[sid]['orientation'], profile['checkpoint_sha256'])
            assert not result['ground_truth_used']
            expected, _, _ = fallback_rows(e, ds, sid, archive)
            fields = ('instance_id', 'text', 'raw_text', 'score', 'bbox', 'boundary_original', 'invalid_envelope')
            assert [[r[k] for k in fields] for r in result['rows']] == [[r[k] for k in fields] for r in expected], sid
            for actual, old in zip(result['rows'], expected):
                assert actual['ignored'] is False
                if 'recognition_confidence' in old:
                    assert abs(actual['recognition_confidence']-old['recognition_confidence']) < 1e-12
            rows_checked += len(expected)
        checks[ds] = dict(images=len(ids), word_rows=rows_checked, checkpoint_sha256=profile['checkpoint_sha256'],
            exact_ids_text_score_geometry=True, ground_truth_used_by_postprocessor=False)
    output = Path(c['output_root'])/'postprocessor_qualification.json'
    if output.exists():
        raise ValueError('Preserve existing fallback qualification')
    write(output, dict(status='PASS', mode='approximate', claim_label='approximate reproduction',
        config_sha256=file_hash(config_path), source_sha256=file_hash(__file__),
        postprocessor_sha256=file_hash(Path(__file__).parents[1]/'recognition/uniform_fallback.py'),
        checks=checks, actual_new_neural_calls=0,
        scope='All1033 authenticated raw TEST responses; exact prediction-only admitted IDs/text/scores/geometry. Cached ignore labels used only in reference audit, never by the deployed postprocessor. No latency or target achievement claim.'))
    print('UNIFORM FALLBACK PARITY PASS', checks, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True); a = p.parse_args()
    try: audit(a.config)
    except BaseException as exc:
        write(Path(read(a.config)['output_root'])/('postprocessor_failure_'+str(time.time_ns())+'.json'),
            dict(status='FAILED', error=repr(exc)))
        raise
