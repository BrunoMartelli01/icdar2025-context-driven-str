from pathlib import Path

import pytest

from scripts.evaluate_totaltext_bert_correction import payload_digest,seal_json,shard_path
from scripts.evaluate_totaltext_bert_visual_correction import ceiling,checked_row,config


def test_candidate_oracle_does_not_count_two_gt_for_one_crop():
    blocks=[{"bbox":[0,0,10,10],"text":"wrong"}]
    gt=[{"bbox":[0,0,10,10],"text":"one"},{"bbox":[0,0,10,10],"text":"two"}]
    assert ceiling(blocks,gt,[[{"text":"one"},{"text":"two"}]])==1
    assert ceiling(blocks,[{"bbox":[20,20,30,30],"text":"one"}],[[{"text":"one"}]])==0


def test_config_namespaces_fail_closed(tmp_path):
    original=Path("configs/approximate/totaltext_bert_visual_correction_v1.json")
    cfg=config(original)
    assert len(cfg["controls"])==6
    dest=tmp_path/"configs"/"strict"/"config.json"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(original.read_bytes())
    with pytest.raises(ValueError,match="approximate"):
        config(dest)


def test_shard_checksum_and_crop_identity_are_bound(tmp_path):
    block={"rank":1,"bbox":[0,0,10,10],"crop_rgb_sha256":"crop","text":"ab","insertion_anchor":"ab"}
    sample={"sample_id":"sid","image_sha256":"rgb","blocks":[block]}
    row={"inputs_sha256":"inputs","runtime_sha256":"runtime","sample_id":"sid","image_sha256":"rgb","trace":[block]}
    row["payload_sha256"]=payload_digest(row)
    file=shard_path(tmp_path,"visual","sid")
    seal_json(file,row)
    assert checked_row(tmp_path,"visual",sample,{"inputs_sha256":"inputs"},"runtime")==row
    sample["blocks"]=[{**block,"bbox":[0,0,11,10]}]
    with pytest.raises(ValueError,match="crop identity"):
        checked_row(tmp_path,"visual",sample,{"inputs_sha256":"inputs"},"runtime")
    sample["blocks"]=[block]
    row["trace"][0]["text"]="ba"
    file.write_text(__import__("json").dumps(row))
    with pytest.raises(ValueError,match="checksum"):
        checked_row(tmp_path,"visual",sample,{"inputs_sha256":"inputs"},"runtime")
