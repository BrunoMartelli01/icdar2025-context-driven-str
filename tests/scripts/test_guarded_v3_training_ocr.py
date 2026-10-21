from collections import Counter
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.train_guarded_v3 import references,guarded_batches
from scripts.run_guarded_v3_training_ocr import resolved_ocr_plan,train_command
from scripts.evaluate_totaltext_compact_v2_selected_ocr import validate_target_variant


def test_validation_references_use_actual_v3_owners_and_keep_absent_ids(tmp_path,monkeypatch):
    path=tmp_path/'images/a.jpg';path.parent.mkdir();Image.new('RGB',(16,16)).save(path)
    owners=np.zeros((128,128),np.int32);owners[3:6,4:8]=1
    item={'native_owners':torch.from_numpy(owners),'mask':torch.from_numpy((owners>0).astype(np.float32)).unsqueeze(0),
        'image_path':str(path),'gt_guard_audit_json':json.dumps({'per_word':[{'owner':1},{'owner':2}]})}
    manifest=SimpleNamespace(samples=[SimpleNamespace(sample_id='total_text:a')])
    monkeypatch.setattr('scripts.train_guarded_v3._gt',lambda *args:[{'text':'A'},{'text':'B'}])
    out=references(manifest,tmp_path,[0],[item],None)['total_text:a']
    np.testing.assert_array_equal(out['owners'],owners);assert out['expected_ids']==[1,2]


def test_guard_logging_keeps_per_sample_evidence_and_batch_unchanged(tmp_path):
    row={'stats':{'words':2,'unresolved_words':1},'per_word':[{'owner':1},{'owner':2}]}
    batch={'target_variant':['guarded-word-targets-v3'],'gt_guard_audit_json':[json.dumps(row)]}
    totals=Counter();path=tmp_path/'audit.jsonl.gz'
    assert list(guarded_batches([batch],path,totals))[0] is batch
    assert totals==row['stats']
    with gzip.open(path,'rt') as f:assert json.loads(f.readline())==row


def test_wrong_target_variant_cannot_enter_v3_training(tmp_path):
    batch={'target_variant':['compact-v2'],'gt_guard_audit_json':[]}
    with pytest.raises(ValueError,match='Non-V3'):list(guarded_batches([batch],tmp_path/'a.gz',Counter()))


@pytest.mark.parametrize('variant,target,recipe,valid',[
    ('guarded-v3-selected-ocr-padding-v1','guarded-word-targets-v3','a',True),
    ('guarded-v3-selected-ocr-padding-v1',None,None,False),
    ('compact-v2-selected-ocr-padding-v1','guarded-word-targets-v3','a',False),
    ('compact-v2-selected-ocr-padding-v1',None,None,True)])
def test_ocr_source_namespaces_are_isolated(variant,target,recipe,valid):
    args=({'variant':variant},{'target_variant':target,'guard_recipe_sha256':recipe})
    if valid:validate_target_variant(*args)
    else:
        with pytest.raises(ValueError):validate_target_variant(*args)


def test_ocr_plan_waits_for_complete_training(tmp_path):
    (tmp_path/'run_manifest.json').write_text(json.dumps({'training_complete':False,'completed_epochs':1,'configured_epochs':500,'test_loaded':False}))
    with pytest.raises(ValueError,match='wait'):resolved_ocr_plan({'ocr_template':{}},tmp_path,0)


def test_resolved_plan_deduplicates_validation_selected_checkpoints(tmp_path):
    source={'training_complete':True,'completed_epochs':500,'configured_epochs':500,'test_loaded':False}
    selection={'native':{'epoch':20},'bbox':{'epoch':20},'iou':{'epoch':10},'last':{'epoch':500}}
    for name,value in [('run_manifest.json',source),('checkpoint_selection.json',selection),('run_contract.json',{}),('epochs.jsonl',{})]:
        (tmp_path/name).write_text(json.dumps(value))
    out=resolved_ocr_plan({'ocr_template':{'variant':'guarded-v3-selected-ocr-padding-v1'}},tmp_path,150)
    assert out['checkpoint_epochs']==[10,20,500] and out['min_unpadded_bbox_area_native']==150
    assert len(out['source_sha256'])==4


def test_preflight_command_is_a_smoke_without_training_resume_flags():
    c={'run_root':'/runs/example','python':'/python','training_config':'configs/approximate/train.json',
        'training_data_root':'/data','encoder_checkpoint':'/encoder','runtime':{'device':'cuda:0'}}
    args=train_command(c,True)
    assert '--check' in args and args[args.index('--check-report')+1].endswith('preflight_training.json')
