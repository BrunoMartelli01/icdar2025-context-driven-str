import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pytest
from PIL import Image

from pipeline.uniform_word_pipeline import input_records, batch_quota
from scripts.run_uniform_unet_training import read, write
from segmentation.uniform_unet_fusion import file_hash


def test_rgb_boundary_records_exact_bytes_dimensions_and_hashes(tmp_path):
    image = np.arange(4*7*3, dtype=np.uint8).reshape(4, 7, 3)
    path = tmp_path/'image.png'; Image.fromarray(image).save(path)
    result = input_records(dict(samples=[dict(sample_id='new:1', image_path=str(path))]))
    assert result[0]['image_wh'] == [7, 4] and result[0]['orientation'] == 1
    assert result[0]['rgb_sha256'] == hashlib.sha256(image.tobytes()).hexdigest()
    assert result[0]['image_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert set(result[0]) == {'sample_id', 'image_path', 'image_wh', 'orientation', 'rgb_sha256', 'image_sha256'}


def test_annotation_fields_and_file_interface_aliases_are_rejected(tmp_path):
    path = tmp_path/'image.png'; Image.new('RGB', (7, 4)).save(path)
    with pytest.raises(ValueError, match='Only sample_id'):
        input_records(dict(samples=[dict(sample_id='one', image_path=str(path), ground_truth=[])]))
    with pytest.raises(ValueError, match='collide'):
        input_records(dict(samples=[dict(sample_id=s, image_path=str(path)) for s in ['a:b', 'a__b']]))
    with pytest.raises(ValueError, match='Nonempty'):
        input_records(dict(samples=[]))
    with pytest.raises(ValueError, match='Only sample_id'):
        input_records(dict(samples=[dict(sample_id='../other', image_path=str(path))]))


def test_reference_budget_is_exact_at_benchmark_n_and_ceil_for_new_batches():
    assert batch_quota(61, 233, 233) == 61
    assert batch_quota(228, 500, 500) == 228
    assert batch_quota(150, 300, 300) == 150
    assert batch_quota(61, 233, 12) == 4
    assert batch_quota(0, 233, 12) == 0
    with pytest.raises(ValueError): batch_quota(234, 233, 12)


def test_fallback_failure_preserves_frozen_rgb_only_route_and_never_completes(tmp_path, monkeypatch):
    # Explicit orchestration fixture: no neural model is executed and no
    # scientific quality/latency result is attributed to these test doubles.
    import pipeline.uniform_word_pipeline as module
    root, output = tmp_path/'training', tmp_path/'output'
    write(root/'d/resolved_configuration.json', dict(partitions=dict(test=['a', 'b', 'c', 'd'])))
    raw_profile, runtime = tmp_path/'neural.json', tmp_path/'runtime.json'
    write(raw_profile, {}); write(runtime, {})
    profile = dict(mode='approximate', claim_label='approximate reproduction', lexicon=None,
        neural_profile=str(raw_profile), runtime=str(runtime), source='fixture-source', python='fixture-python',
        checkpoint_sha256='fixture-checkpoint', policy=dict(decoder='greedy', same_text_nms_iou=None,
            resolution='native', score_threshold=.4, recognition_threshold=0.))
    fallback = tmp_path/'fallback.json'
    write(fallback, dict(output_root=str(tmp_path/'qualification'), profiles=dict(d=profile),
        pinned_inputs={str(p):file_hash(p) for p in [raw_profile, runtime]}))
    write(tmp_path/'qualification/postprocessor_qualification.json', dict(status='PASS', config_sha256=file_hash(fallback),
        postprocessor_sha256=file_hash(Path(module.__file__).parents[1]/'recognition/uniform_fallback.py')))
    config = tmp_path/'approximate/config.json'
    write(config, dict(mode='approximate', pinned_inputs={}, fallback_profiles=str(fallback), utility_config='fixture',
        router_cpu_threads=1, minimum_free_fallback_gpu_bytes=1))
    image = tmp_path/'image.png'; Image.new('RGB', (7, 4)).save(image)
    manifest = tmp_path/'images.json'
    write(manifest, dict(samples=[dict(sample_id=s, image_path=str(image)) for s in ['first', 'second']]))

    class FixtureUtility:
        def __init__(self, *args):
            self.config = dict(training_root=str(root)); self.policy = dict(guard=False)
        def observe(self, *args): return dict(score=0., rows=[])

    def fixture_frontend(training_root, records, folder):
        index = {}
        for row in records:
            mp = folder/(row['sample_id']+'.npz'); mp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mp, probabilities=np.zeros((2, 4, 4)))
            sample = folder/(row['sample_id']+'.json')
            write(sample, dict(ground_truth_used=False, maps_path=str(mp), maps_sha256=file_hash(mp),
                raw_rows=[], image_wh=row['image_wh']))
            index[row['sample_id']] = dict(path=str(sample), sha256=file_hash(sample))
        return index, dict(actual_unet_image_inferences=0, actual_parseq_crop_inferences=0)

    def failed_worker(command, **kwargs):
        assert command[0] == 'fixture-python' and kwargs['check']
        decision = read(output/'decisions_before_fallback.json')
        assert decision['status'] == 'FROZEN' and decision['deepsolo_images'] == 1
        assert not decision['ground_truth_used'] and not decision['fallback_outputs_loaded']
        export = read(output/'fallback_plan.json')
        selected = export['population']['test']['d']
        assert len(selected) == 1 and set(selected[0]) == {'sample_id', 'image_path', 'image_sha256'}
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(module, 'UniformUtilityPipeline', FixtureUtility)
    monkeypatch.setattr(module, 'frontend', fixture_frontend)
    monkeypatch.setattr(module.subprocess, 'run', failed_worker)
    with pytest.raises(subprocess.CalledProcessError):
        module.run(config, manifest, 'd', 2, output)
    assert (output/'decisions_before_fallback.json').exists()
    assert not (output/'complete.json').exists() and not (output/'predictions.json').exists()
