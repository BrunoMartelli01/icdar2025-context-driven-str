from copy import deepcopy
from pathlib import Path
import json

import pytest

import segmentation.text_recall_provenance as provenance
from scripts.train_totaltext_v4_highres_ablation import _json


@pytest.fixture
def failed_run(tmp_path):
    current={"source_sha256":{provenance.RUNNER:provenance.CORRECTED_RUNNER_SHA256,provenance.PROVENANCE:'new-provenance',
        provenance.CUDA_HELPER:provenance.QUALIFIED_CUDA_HELPER_SHA256,'segmentation/text_recall.py':'unchanged-science'},
        "base_checkpoint_sha256":'fixed-base',"config":{"base_checkpoint_epoch":56,
        "architecture_identity":'practical-totaltext-v4-highres-agunet-v1',
        "training":{"batch_size":4},"dataset":{"images":1}}}
    previous=provenance.legacy_contract(current)
    stub={**current,'branch':'control512','training_config':{'loss':'masked_BCE_Dice'},'execution_device':'cuda:0'}
    old_stub={**previous,'branch':'control512','training_config':stub['training_config'],'execution_device':'cuda:0'}
    historical={'reference':'immutable'}
    _json(tmp_path/'experiment_contract.json',previous)
    _json(tmp_path/'historical_reference.json',historical)
    _json(tmp_path/'target_qualification.json',{'status':'PASS','result_class':'smoke-test result','augmentation':'disabled',
        'run_contract':previous,'resolutions':{str(size):{'images':1,'per_image':[{'sample_id':'word:1'}]} for size in (512,768)}})
    _json(tmp_path/provenance.STUB,old_stub)
    _json(tmp_path/'smoke/cuda_0/control512/warm_start.json',{'source_checkpoint_sha256':'fixed-base',
        'source_epoch':56,'architecture':current['config']['architecture_identity'],'parameters':58155081})
    _json(tmp_path/'failure-1.json',{'status':'FAILED','result_class':'smoke-test result','action':'smoke',
        'branch':'control512','device':'cuda:0','type':'RuntimeError','message':'Invalid device argument ',
        'oom':False,'precision':'float32','peak_allocated_bytes':0,'peak_reserved_bytes':0,
        'batch_size':4,'input_shape':[4,3,512,512]})
    return tmp_path,current,historical,stub


def test_known_failure_archives_bytes_and_reuses_preflight_idempotently(failed_run):
    root,current,historical,stub=failed_run
    originals={name:(root/name).read_bytes() for name in ('experiment_contract.json','target_qualification.json',provenance.STUB)}
    failure=(root/'failure-1.json').read_bytes()
    provenance.ensure_experiment_contract(root,current,historical,stub)
    for name,data in originals.items():assert (root/provenance.ARCHIVE/name).read_bytes()==data
    assert (root/'failure-1.json').read_bytes()==failure
    assert json.loads((root/'experiment_contract.json').read_text())==current
    saved=json.loads((root/'target_qualification.json').read_text())
    assert saved['resolutions']==json.loads(originals['target_qualification.json'])['resolutions']
    all_bytes={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    provenance.ensure_experiment_contract(root,current,historical,stub)
    assert all((root/name).read_bytes()==data for name,data in all_bytes.items())


@pytest.mark.parametrize('name',['branches/control512/best.pt','smoke/cuda_0/control512/smoke.json',
    'branches/control512/epochs.jsonl','unexpected.json'])
def test_completed_or_unknown_artifacts_refuse_without_modification(failed_run,name):
    root,current,historical,stub=failed_run
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('do not modify')
    before=(root/'experiment_contract.json').read_bytes()
    with pytest.raises(ValueError):provenance.ensure_experiment_contract(root,current,historical,stub)
    assert (root/'experiment_contract.json').read_bytes()==before and not (root/provenance.ARCHIVE).exists()


@pytest.mark.parametrize('field', ['scientific_source','configuration','failure','preflight','warm_start','historical',
                                 'unqualified_runner','unqualified_cuda_helper'])
def test_scientific_change_or_bad_evidence_cannot_enter_repair(failed_run,field):
    root,current,historical,stub=failed_run
    current=deepcopy(current)
    if field=='scientific_source':current['source_sha256']['segmentation/text_recall.py']='changed'
    elif field=='configuration':current['config']['training']['batch_size']=2
    elif field=='unqualified_runner':current['source_sha256'][provenance.RUNNER]='unqualified'
    elif field=='unqualified_cuda_helper':current['source_sha256'][provenance.CUDA_HELPER]='unqualified'
    else:
        name={'failure':'failure-1.json','preflight':'target_qualification.json',
              'warm_start':'smoke/cuda_0/control512/warm_start.json','historical':'historical_reference.json'}[field]
        data=json.loads((root/name).read_text())
        if field=='failure':data['message']='different runtime error'
        elif field=='preflight':data['status']='FAILED'
        elif field=='warm_start':data['source_checkpoint_sha256']='different'
        else:data['reference']='different'
        _json(root/name,data)
    before={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):provenance.ensure_experiment_contract(root,current,historical,stub)
    assert all((root/name).read_bytes()==data for name,data in before.items())
    assert not (root/provenance.ARCHIVE).exists()


def test_interrupted_metadata_commit_resumes_from_archived_originals(failed_run,monkeypatch):
    root,current,historical,stub=failed_run;original_writer=provenance._json
    def interrupt(path,data):
        if path==root/provenance.STUB:raise RuntimeError('interrupt between metadata commits')
        original_writer(path,data)
    monkeypatch.setattr(provenance,'_json',interrupt)
    with pytest.raises(RuntimeError):provenance.ensure_experiment_contract(root,current,historical,stub)
    assert json.loads((root/'experiment_contract.json').read_text())==provenance.legacy_contract(current)
    assert json.loads((root/'target_qualification.json').read_text())['run_contract']==current
    monkeypatch.setattr(provenance,'_json',original_writer)
    provenance.ensure_experiment_contract(root,current,historical,stub)
    assert json.loads((root/provenance.STUB).read_text())==stub


def test_modified_archive_is_not_silently_accepted(failed_run):
    root,current,historical,stub=failed_run
    provenance.ensure_experiment_contract(root,current,historical,stub)
    path=root/provenance.ARCHIVE/'experiment_contract.json';path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='checksum'):provenance.ensure_experiment_contract(root,current,historical,stub)
