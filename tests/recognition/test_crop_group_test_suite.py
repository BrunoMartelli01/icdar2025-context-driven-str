import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from localize.crop_recovery import mask_candidates
from scripts import run_totaltext_crop_group_test_suite as runner
from scripts.analyze_totaltext_crop_word_groups import format_optional, validate_config as validate_groups


def test_test_configuration_freezes_train_rules_without_final_variations():
    cfg = runner.read_config(runner.CONFIG)
    train = json.loads((runner.REPO / 'configs/approximate/totaltext_crop_recovery_suite_train_v1.json').read_text())
    for key in ('segmentation_checkpoint_sha256', 'baseline_threshold', 'baseline_padding',
                'mask_size', 'matching_iou', 'normalization', 'threshold_tree', 'tiled_segmentation'):
        assert cfg[key] == train[key]
    group = json.loads(runner.GROUP_CONFIG.read_text())
    validate_groups(group)
    original = json.loads((runner.REPO / 'configs/approximate/totaltext_crop_word_groups_train_v1.json').read_text())
    assert group['strategies'] == ['complete_link'] and group['iou_min_values'] == [.55]
    for key in ('center_limit', 'ambiguity', 'evaluation', 'source_methods'):
        assert group[key] == original[key]
    assert format_optional(None, 2) == 'n/a'  # Images can have no recoverable OCR word.
    assert format_optional(0., 4) == '0.0000'


def test_test_manifest_role_count_hash_and_limit_are_enforced(tmp_path, monkeypatch):
    path = tmp_path / 'manifest.yaml'; path.write_text('fixture')
    cfg = {'test_manifest_sha256': runner.sha(path), 'test_images': 2}
    manifest = SimpleNamespace(dataset='total_text', split_role='train', samples=[1,2])
    monkeypatch.setattr(runner, 'load_segmentation_manifest', lambda *a, **k: manifest)
    with pytest.raises(ValueError, match='TRAIN manifests'):
        runner.test_samples(path, cfg)
    manifest.split_role='test'
    assert runner.test_samples(path, cfg) == [1,2]
    assert runner.test_samples(path, cfg, 1) == [1]
    with pytest.raises(ValueError, match='limit'):
        runner.test_samples(path, cfg, 3)
    path.write_text('changed')
    with pytest.raises(ValueError, match='hash'):
        runner.test_samples(path, cfg)


def test_threshold_only_path_preserves_original_threshold_crops():
    cfg = runner.read_config(runner.CONFIG)
    p = np.zeros((256,256), np.float32)
    p[70:95, 50:90] = .95; p[70:95, 110:155] = .95
    p[82:84, 90:110] = .55
    image = np.zeros((256,256,3), np.uint8)
    whole = mask_candidates(p, image, cfg)
    subset = mask_candidates(p, image, cfg, include_auxiliary=False)
    keep = {'baseline','threshold_tree'}
    original = {tuple(r['bbox']) for r in whole.rows.values() if set(r['methods']) & keep}
    assert {tuple(r['bbox']) for r in subset.rows.values()} == original
    assert all(set(r['methods']) <= keep and r['quad'] is None and r['neutral_fraction']==0
               for r in subset.rows.values())


def test_source_evaluation_is_exact_once_and_test_denominator_checked(tmp_path, monkeypatch):
    sample=SimpleNamespace(sample_id='total_text:1')
    image=np.zeros((10,10,3),np.uint8)
    monkeypatch.setattr(runner, '_image', lambda *a: image)
    monkeypatch.setattr(runner, '_ground_truth', lambda *a: ([{'bbox':[0,0,10,10],'text':'WORD!'}],'annotation-sha'))
    rows=[{'candidate_id':'a','bbox':[0,0,10,10],'crop_rgb_sha256':'a','methods':['baseline','threshold_tree']},
          {'candidate_id':'b','bbox':[0,0,10,10],'crop_rgb_sha256':'b','methods':['threshold_tree']}]
    prepared={sample.sample_id:{'candidates':rows}}
    readings={sample.sample_id:{'a':{'raw_text':'word'},'b':{'raw_text':'Word$'}}}
    for directory in ['prepared','ocr']:
        (tmp_path/directory).mkdir(); (tmp_path/directory/'total_text__1.json').write_text('{}')
    (tmp_path/'resolved_configuration.json').write_text('{}')
    cfg=runner.read_config(runner.CONFIG); cfg['test_care_words']=1
    runner.evaluate_source([sample], prepared, readings, {'data_root':tmp_path}, cfg, tmp_path, False)
    result=json.loads((tmp_path/'results.json').read_text())
    assert result['summary']['all_methods']['correct']==1
    assert result['summary']['all_methods']['candidates']==2
    assert result['result_class']=='APPROXIMATE_TEST_DIAGNOSTIC'
    cfg['test_care_words']=2
    with pytest.raises(RuntimeError, match='denominator'):
        runner.evaluate_source([sample], prepared, readings, {'data_root':tmp_path}, cfg, tmp_path, False)


def test_complete_test_source_reuse_rejects_drifted_inputs(tmp_path):
    source=tmp_path/'source'
    source.mkdir()
    config_path=tmp_path/'config.json'; config_path.write_text('{}')
    manifest_path=tmp_path/'test.yaml'; manifest_path.write_text('test')
    sample=SimpleNamespace(sample_id='total_text:1')
    local={field:tmp_path/field for field in
           ('data_root','training_runs','parseq_source','parseq_archive','parseq_checkpoint')}
    plan={'reproduction_mode':'approximate','config':{'split':'official_TEST_300'},
          'config_sha256':runner.sha(config_path),'manifest_sha256':runner.sha(manifest_path),
          'sample_ids':[sample.sample_id],'execution':{'device':'cuda:0'},
          'paths':{field:str(path) for field,path in local.items()},
          'code_sha256':{name:runner.sha(runner.REPO/name) for name in runner.crops.CODE_FILES}}
    plan['code_sha256']['scripts/run_totaltext_crop_group_test_suite.py']=runner.REUSABLE_SOURCE_RUNNER_SHA256
    (source/'resolved_configuration.json').write_text(json.dumps(plan))
    result={'plan_sha256':runner.sha(source/'resolved_configuration.json'), 'images':1,
            'summary':{'baseline':{'gt_words':1}},'table_5_claim':False}
    (source/'results.json').write_text(json.dumps(result))
    for name in ('prepared','ocr','evaluation','tiles'):
        (source/name).mkdir(); (source/name/'total_text__1.json').write_text('{}')
    cfg={'test_care_words':1}
    assert runner.qualify_reusable_source(source,cfg,config_path,manifest_path,
        [sample],'cuda:0',local)==plan
    with pytest.raises(RuntimeError,match='contract'):
        runner.qualify_reusable_source(source,cfg,config_path,manifest_path,
            [sample],'cpu',local)
    plan['code_sha256']['localize/crop_recovery.py']='0'*64
    (source/'resolved_configuration.json').write_text(json.dumps(plan))
    result['plan_sha256']=runner.sha(source/'resolved_configuration.json')
    (source/'results.json').write_text(json.dumps(result))
    with pytest.raises(RuntimeError,match='dependency changed'):
        runner.qualify_reusable_source(source,cfg,config_path,manifest_path,
            [sample],'cuda:0',local)
