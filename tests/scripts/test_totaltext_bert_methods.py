"""Frozen method-battery and caption boundaries; pretrained scores are not simulated."""
import argparse
import json
from pathlib import Path

import pytest
import torch
from transformers import BertConfig,BertForMaskedLM

import scripts.evaluate_totaltext_bert_methods as battery
from scripts.generate_totaltext_bert_captions import bind_rgb, caption_path, checked_caption
from test_totaltext_neuspell import experiment as upstream_experiment


CONFIG=Path(__file__).resolve().parents[2]/"configs/approximate/totaltext_bert_methods_v1.json"


@pytest.fixture
def experiment(upstream_experiment):
    args=upstream_experiment
    upstream=battery.read_json(args.config)
    cfg=battery.read_json(CONFIG)
    for key in ("source","model","upstream_neuspell"):
        cfg[key]=upstream[key]
    cfg.pop("assets_evidence",None)
    cfg["scorer"].update(model_id="fixture/tiny-mlm",revision="e"*40,max_input_tokens=32)
    snapshot=battery.read_json(args.asset_dir/"assets.json")
    head_root=args.hf_cache/("models--"+cfg["model"]["model_id"].replace("/","--"))/"snapshots"/cfg["model"]["revision"]
    # Match the legacy lowercasing control; an explicit cased variant is tested separately.
    tok=battery.read_json(head_root/"tokenizer_config.json")
    tok["do_lower_case"]=True
    (head_root/"tokenizer_config.json").write_text(json.dumps(tok))
    snapshot["model_files_sha256"]["tokenizer_config.json"]=battery.digest(head_root/"tokenizer_config.json")
    (args.asset_dir/"assets.json").write_text(json.dumps(snapshot))
    # Use an OOV source, so the hybrid reranker actually executes its genuine MLM.
    vocabulary=battery.read_json(args.asset_dir/"vocabulary.json")
    import pickle
    from recognition.neuspell_correction import git_blob_digest,convert_vocabulary
    labels=vocabulary["labels"]
    labels[5]="older"
    raw=pickle.dumps(dict(idx2token=dict(enumerate(labels)),token2idx={v:i for i,v in enumerate(labels)},
        pad_token=labels[0],unk_token=labels[1]))
    (args.asset_dir/"vocab.pkl").write_bytes(raw)
    cfg["upstream_neuspell"]["vocabulary_git_blob"]=git_blob_digest(raw)
    vocabulary=convert_vocabulary(args.asset_dir/"vocab.pkl",expected_blob=git_blob_digest(raw),expected_labels=6)
    (args.asset_dir/"vocabulary.json").write_text(json.dumps(vocabulary))
    snapshot["upstream_neuspell"]=cfg["upstream_neuspell"]
    for n in ("vocab.pkl","vocabulary.json"):
        snapshot["auxiliary_files_sha256"][n]=battery.digest(args.asset_dir/n)
    (args.asset_dir/"assets.json").write_text(json.dumps(snapshot))
    mlm_root=args.hf_cache/"models--fixture--tiny-mlm"/"snapshots"/("e"*40)
    mlm_root.mkdir(parents=True)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(head_root,local_files_only=True,use_fast=True)
    tokenizer.save_pretrained(mlm_root)
    model=BertForMaskedLM(BertConfig(vocab_size=len(tokenizer),hidden_size=16,num_hidden_layers=1,
        num_attention_heads=2,intermediate_size=32,max_position_embeddings=64))
    with torch.no_grad():
        model.cls.predictions.decoder.weight.zero_()
        model.cls.predictions.bias.fill_(-10)
        model.cls.predictions.bias[tokenizer.convert_tokens_to_ids("parking")]=6
    model.save_pretrained(mlm_root,safe_serialization=True)
    cfg["scorer"]["weight_sha256"]=battery.digest(mlm_root/"model.safetensors")
    args.config.write_text(json.dumps(cfg))
    args.output_dir=args.output_dir.parent/"battery-output"
    args.conditions=["word_only","local_ocr"]
    args.captions=None
    battery.prepare(args)
    return args


def test_battery_real_tiny_heads_pll_and_evaluation_without_gt_then_resume(experiment):
    args=experiment
    out=args.output_dir
    files={n:(out/n).read_bytes() for n in ("evaluation_labels.json","baseline.json")}
    for n in files:
        (out/n).unlink()
    battery.infer_head(args)
    battery.infer_pll(args)
    for n,data in files.items():
        (out/n).write_bytes(data)
    battery.evaluate(args)
    result=battery.read_json(out/"results.json")
    assert len(result["conditions"])==12 and not result["scene_context_evaluated"]
    for row in result["conditions"].values():
        assert row["recovered"]==1 and row["damaged"]==0
        assert row["metrics"]["recognition"]["exact_casefold_rate_all_gt"]==.5
        assert row["metrics"]["geometry"]==result["baseline"]["geometry"]
    paths=list((out/"head").rglob("*.json"))+list((out/"pll").rglob("*.json"))
    before={str(p):battery.digest(p) for p in paths}
    battery.infer_head(args)
    battery.infer_pll(args)
    assert before=={str(p):battery.digest(p) for p in paths}
    cfg,record,inputs=battery.prepared(out)
    sample=inputs["samples"][0]
    path=battery.head_path(out,"published_lowercase","local_ocr",sample["sample_id"])
    row=battery.read_json(path)
    row["trace"][0]["context"]["left"]=["injected"]
    row["payload_sha256"]=battery.payload_digest({k:v for k,v in row.items() if k != "payload_sha256"})
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match="context differs"):
        battery.evaluate(args)


def test_full_scene_condition_rejects_missing_captions_and_wrong_model(experiment):
    args=experiment
    args.output_dir=args.output_dir.parent/"scene-battery"
    args.conditions=None
    with pytest.raises(ValueError,match="complete SHA-bound"):
        battery.prepare(args)
    args.captions=args.output_dir.parent/"wrong-captions.json"
    args.captions.write_text(json.dumps(dict(model_id="wrong",revision="f"*40,generation={},model_files_sha256={})))
    with pytest.raises(ValueError,match="Caption model"):
        battery.prepare(args)


def test_caption_rgb_binding_rejects_different_image_and_annotation_is_not_opened(tmp_path):
    import yaml
    image=tmp_path/"rgb.jpg"
    image.write_bytes(b"test-only-rgb-file")
    inputs=dict(samples=[dict(sample_id="total_text:14",image_sha256=battery.digest(image))])
    manifest=tmp_path/"rgb-manifest.yaml"
    obj=dict(dataset="total_text",split_role="train",samples=[dict(id="total_text:14",image="rgb.jpg",
        image_sha256=battery.digest(image),annotation="nonexistent-label-file")])
    manifest.write_text(yaml.safe_dump(obj))
    bound=bind_rgb(inputs,manifest,tmp_path,battery.digest(manifest))
    assert len(bound)==1
    image.write_bytes(b"different")
    with pytest.raises(ValueError,match="frozen OCR"):
        bind_rgb(inputs,manifest,tmp_path,battery.digest(manifest))


def test_caption_shard_corruption_and_coverage_binding(tmp_path):
    sample=dict(sample_id="total_text:14",image_sha256="a"*64)
    row={**sample,"runtime_sha256":"b"*64,"text":"A sign outside a shop."}
    row["payload_sha256"]=battery.payload_digest(row)
    path=caption_path(tmp_path,sample["sample_id"])
    battery.seal_json(path,row)
    assert checked_caption(path,sample,"b"*64)["text"]==row["text"]
    row["text"]="injected"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match="caption shard"):
        checked_caption(path,sample,"b"*64)
