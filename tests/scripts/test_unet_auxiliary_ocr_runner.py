import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_unet_auxiliary_ocr import read_config, json_write, split_population


def test_explicit_namespace_and_valid_config(tmp_path):
    real=Path(__file__).resolve().parents[2]/'configs/approximate/unet_auxiliary_ocr_v1.json'
    c=read_config(real,'approximate')
    assert c['candidate_epochs']==[0,5,10,20] and c['ocr_weight']>0
    other=tmp_path/'x.json';other.write_text(real.read_text())
    with pytest.raises(ValueError,match='namespace'):read_config(other,'approximate')
    with pytest.raises(ValueError):read_config(real,'strict')


def test_immutable_receipt_rejects_mutation(tmp_path):
    p=tmp_path/'x.json'
    json_write(p,{'epoch':1},immutable=True);json_write(p,{'epoch':1},immutable=True)
    with pytest.raises(ValueError):json_write(p,{'epoch':2},immutable=True)
    assert json.loads(p.read_text())=={'epoch':1}


def test_source_split_preserved_and_calibration_disjoint():
    from scripts.train_totaltext_v4_highres_ablation import _split_indices
    m=SimpleNamespace(samples=[SimpleNamespace(sample_id=str(i)) for i in range(1255)])
    source=dict(seed=202503,validation_count=125)
    source['validation_ids_sha256']=_split_indices(m,202503,125)[2]
    c=dict(split_seed=20261020,fit_images=1030,calibration_images=100,evaluation_images=125)
    result=split_population(m,source,c)
    assert [len(result[k]) for k in ['fit','calibration','evaluation']]==[1030,100,125]
    assert len(set().union(*map(set,result.values())))==1255
    assert result==split_population(m,source,c)


@pytest.mark.parametrize('field,value',[('lambda_text',-1.),('smooth',0.),('separation_weight',float('nan'))])
def test_invalid_scientific_loss_rejected_before_training(tmp_path,monkeypatch,field,value):
    import scripts.run_unet_auxiliary_ocr as runner
    real=Path(__file__).resolve().parents[2]/'configs/approximate/unet_auxiliary_ocr_v1.json'
    c=json.loads(real.read_text());c['loss'][field]=value
    p=tmp_path/'configs/approximate/config.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(c))
    monkeypatch.setattr(runner,'REPO',tmp_path)
    with pytest.raises(ValueError,match='loss weights'):runner.read_config(p,'approximate')
