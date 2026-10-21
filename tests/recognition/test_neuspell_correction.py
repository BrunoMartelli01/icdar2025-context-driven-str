"""NeuSpell decoding contracts. Tiny random weights are engineering fixtures."""

import math
import pickle

import pytest
import torch
from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

from recognition.neuspell_correction import (
    NeuSpellWordHead, SpellPolicy, convert_vocabulary, correct_blocks,
    git_blob_digest, load_word_head, select_spelling,
)


def policy(**changes):
    return SpellPolicy(**{**dict(min_word_length=3, long_word_length=6, short_max_edits=1,
        long_max_edits=2, max_relative_edit=.3, minimum_probability=.5,
        minimum_log_probability_gain=1.5, neighbours_per_side=3, min_vertical_overlap=.5,
        max_horizontal_gap_heights=8, max_caption_words=80), **changes})


def vocabulary():
    labels = ["<<PAD>>", "<<UNK>>", "parking", "PARKING", "hotel", "parkin"]
    return {"idx2token": dict(enumerate(labels)), "token2idx": {v: i for i, v in enumerate(labels)},
            "pad_token": labels[0], "unk_token": labels[1]}


@pytest.fixture
def head(tmp_path):
    path = tmp_path / "vocab.txt"
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "park", "##in", "parking",
              "PARKING", "hotel", "motel", "the", "is", "sign", ".", "seven", "shop"]
    path.write_text("\n".join(tokens) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(path), do_lower_case=False)
    torch.manual_seed(17)
    model = BertForTokenClassification(BertConfig(vocab_size=len(tokens), hidden_size=16,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        max_position_embeddings=64, num_labels=6)).eval()
    return NeuSpellWordHead(tokenizer, model,
        {"labels": list(vocabulary()["idx2token"].values()), "special_ids": [0, 1]},
        device="cpu", batch_size=2, max_input_tokens=32, saved_top_k=6)


def request(original="parkin", left=(), right=(), caption=None):
    return dict(original=original, left=left, right=right, caption=caption)


def evidence(proposed="hotel", probability=.8, gain=2, oov=False, **changes):
    return {**dict(status="scored", prediction=proposed, probability=probability,
        log_probability=math.log(probability), special_prediction=False,
        reference_log_probability=None if oov else math.log(probability) - gain), **changes}


def test_authenticated_vocabulary_preserves_exact_class_order(tmp_path):
    raw = pickle.dumps(vocabulary())
    path = tmp_path / "vocab.pkl"
    path.write_bytes(raw)
    obj = convert_vocabulary(path, expected_blob=git_blob_digest(raw), expected_labels=6)
    assert obj["labels"] == list(vocabulary()["idx2token"].values())
    assert obj["special_ids"] == [0, 1]
    with pytest.raises(ValueError, match="blob hash"):
        convert_vocabulary(path, expected_blob="0" * 40, expected_labels=6)
    corrupted = vocabulary()
    corrupted["token2idx"]["parking"] = 3
    path.write_bytes(pickle.dumps(corrupted))
    with pytest.raises(ValueError, match="inconsistent"):
        convert_vocabulary(path, expected_blob=git_blob_digest(path.read_bytes()), expected_labels=6)


def test_vocabulary_pickle_rejects_executable_objects(tmp_path):
    # A builtin global is rejected without executing or importing it.
    path = tmp_path / "vocab.pkl"
    path.write_bytes(pickle.dumps({"payload": eval}))
    with pytest.raises(pickle.UnpicklingError, match="executable/global"):
        convert_vocabulary(path, expected_blob=git_blob_digest(path.read_bytes()), expected_labels=6)


def test_mean_subword_head_matches_author_reference_with_padding(head):
    requests = [request(), request(left=["the", "shop"], right=["sign"], caption="seven parking")]
    outputs = head.predict(requests)
    tasks = [head.encode_target(r["original"], r["left"], r["right"], r["caption"])[0] for r in requests]
    batch = head.tokenizer.pad([{k: t[k] for k in ("input_ids", "attention_mask")} for t in tasks],
                               return_tensors="pt")
    with torch.inference_mode():
        hidden = head.model.bert(**batch).last_hidden_state
        for i, task in enumerate(tasks):
            positions = task["positions"]
            assert len(positions) == 2  # park + ##in
            reference = (hidden[i, positions[0]] + hidden[i, positions[1]]) / 2
            probabilities = head.model.classifier(reference).log_softmax(-1)
            winner = int(probabilities.argmax())
            assert outputs[i]["label_id"] == winner
            assert outputs[i]["log_probability"] == pytest.approx(float(probabilities[winner]), abs=1e-6)
            assert outputs[i]["reference_log_probability"] == pytest.approx(float(probabilities[5]), abs=1e-6)
    assert head.input_shapes == {tuple(batch["input_ids"].shape)}
    assert outputs == head.predict(requests)


def test_visible_original_caption_target_and_casefold_keep_aliases(head):
    task, observable = head.encode_target("parkin", ["the"], ["sign"], "seven parking")
    assert observable["encoder_text"] == "seven parking. the parkin sign"
    start, end = observable["target_character_span"]
    assert observable["encoder_text"][start:end] == "parkin"
    assert head.tokenizer.mask_token_id not in task["input_ids"]
    assert observable["target_subword_ids"] == head.tokenizer.encode("parkin", add_special_tokens=False)
    out = head.predict([request("parking"), request("motel")])
    assert out[0]["reference_label_ids"] == [2, 3]
    p = sum(r["probability"] for r in out[0]["top_labels"] if r["label_id"] in [2, 3])
    assert out[0]["reference_probability"] == pytest.approx(p, rel=1e-6)
    assert out[1]["reference_label_ids"] == []
    assert out[1]["reference_probability"] is None


def test_argmax_ties_match_author_first_index_and_special_keep(head):
    with torch.no_grad():
        head.model.classifier.weight.zero_()
        head.model.classifier.bias.zero_()
    row = head.predict([request()])[0]
    assert row["label_id"] == 0 and row["special_prediction"]
    decisions = select_spelling("parkin", row, policy())
    assert all(d["text"] == "parkin" for d in decisions.values())


def test_unknown_and_overflow_are_explicit_abstentions_and_reset_shapes(head):
    out = head.predict([request("qzx"), request(left=["the"] * 32)])
    assert [r["status"] for r in out] == ["unknown_or_special_subword", "input_token_limit"]
    assert head.input_shapes == set()
    head.predict([request()])
    assert head.input_shapes
    blocks = [{"rank": 1, "bbox": [0, 0, 10, 10], "crop_rgb_sha256": "a" * 64, "text": "123"}]
    trace = correct_blocks(blocks, condition="word_only", caption=None, decoder=head, policy=policy())
    assert head.input_shapes == set()
    assert trace[0]["neural"]["status"] == "protected_word_shape"


def test_nonfinite_head_and_runtime_failures_propagate(head, monkeypatch):
    with torch.no_grad():
        head.model.classifier.bias[2] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        head.predict([request()])
    def fail(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("test-only OOM")
    monkeypatch.setattr(head.model.bert, "forward", fail)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        head.predict([request()])


@pytest.mark.parametrize("original,row,expected,reason", [
    ("MOTEL", evidence(), "HOTEL", "in_vocab_gain"),
    ("Motel", evidence(oov=True), "Hotel", "oov_reference_probability_gate"),
    ("motel", evidence(probability=.5), "hotel", "in_vocab_gain"),
    ("motel", evidence(probability=.499), "motel", "probability_bound_KEEP"),
    ("motel", evidence(gain=1.5), "motel", "reference_gain_KEEP"),
    ("motel", evidence(proposed="parking"), "motel", "edit_bound_KEEP"),
    ("motel", evidence(proposed="hotel!"), "motel", "candidate_shape_KEEP"),
    ("HOTEL", evidence(), "HOTEL", "same_word_KEEP"),
])
def test_conservative_boundaries_case_and_oov(original, row, expected, reason):
    result = select_spelling(original, row, policy())
    assert result["native_head"]["text"] == row["prediction"]
    assert result["conservative"]["text"] == expected
    assert result["conservative"]["reason"] == reason


def test_context_stays_original_and_crops_stay_bound(head):
    with torch.no_grad():
        head.model.classifier.weight.zero_()
        head.model.classifier.bias.zero_()
        head.model.classifier.bias[2] = 8
    texts = ["parkin", "hotel", "iPhone", "abc!"]
    blocks = [dict(rank=i, bbox=[i * 20, 0, i * 20 + 15, 10],
                   crop_rgb_sha256=str(i) * 64, text=t) for i, t in enumerate(texts)]
    result = correct_blocks(blocks, condition="local_ocr_caption", caption="seven parking",
                            decoder=head, policy=policy())
    assert result[1]["context"]["left"] == ["parkin"]
    assert result[0]["selections"]["conservative"]["text"] == "parking"
    assert all(result[i]["neural"]["status"] == "protected_word_shape" for i in [2, 3])
    for before, after in zip(blocks, result):
        assert all(before[k] == after[k] for k in ["rank", "bbox", "crop_rgb_sha256"])


@pytest.mark.parametrize("mutation,expected", [
    ("missing_head", "compatibility"), ("wrong_shape", "compatibility"),
    ("unapproved_extra", "compatibility"), ("not_tensor", "tensor-only"),
])
def test_scientific_checkpoint_loading_fails_closed(head, tmp_path, mutation, expected):
    head.model.config.save_pretrained(tmp_path)
    state = head.model.state_dict()
    if mutation == "missing_head":
        del state["classifier.weight"]
    elif mutation == "wrong_shape":
        state["classifier.weight"] = state["classifier.weight"][:1]
    elif mutation == "unapproved_extra":
        state["other_head.weight"] = torch.zeros(1)
    else:
        state["classifier.bias"] = "invalid"
    torch.save(state, tmp_path / "pytorch_model.bin")
    with pytest.raises((ValueError, RuntimeError), match=expected):
        load_word_head(tmp_path, device="cpu", expected_labels=6, allowed_unused_keys=[])


def test_only_explicit_dead_pooler_keys_are_audited_and_excluded(head, tmp_path):
    head.model.config.save_pretrained(tmp_path)
    state = head.model.state_dict()
    state["bert.pooler.dense.weight"] = torch.zeros(16, 16)
    torch.save(state, tmp_path / "pytorch_model.bin")
    model, audit = load_word_head(tmp_path, device="cpu", expected_labels=6,
                                 allowed_unused_keys=["bert.pooler.dense.weight"])
    assert audit["unused_keys"] == ["bert.pooler.dense.weight"]
    assert audit["load_state_dict_strict"] and not audit["missing_keys"]
    assert torch.equal(model.classifier.weight, head.model.classifier.weight)


def test_equivalent_legacy_position_buffer_is_verified_before_strict_load(head, tmp_path):
    head.model.config.save_pretrained(tmp_path)
    state = head.model.state_dict()
    state["bert.embeddings.position_ids"] = head.model.bert.embeddings.position_ids.clone()
    torch.save(state, tmp_path / "pytorch_model.bin")
    model, audit = load_word_head(tmp_path, device="cpu", expected_labels=6,
                                  allowed_unused_keys=["bert.embeddings.position_ids"])
    buffer = audit["verified_regenerated_buffers"]["bert.embeddings.position_ids"]
    assert buffer == {"shape": [1, 64], "dtype": "torch.int64", "exact_equality_with_current_buffer": True}
    assert audit["load_state_dict_strict"] and not audit["missing_keys"]
    assert torch.equal(model.classifier.weight, head.model.classifier.weight)
    with pytest.raises(RuntimeError, match="compatibility"):
        load_word_head(tmp_path, device="cpu", expected_labels=6, allowed_unused_keys=[])


@pytest.mark.parametrize("mutation", ["value", "shape", "dtype"])
def test_non_equivalent_legacy_position_buffer_is_never_discarded(head, tmp_path, mutation):
    head.model.config.save_pretrained(tmp_path)
    state = head.model.state_dict()
    positions = head.model.bert.embeddings.position_ids.clone()
    if mutation == "value":
        positions[0, 3] = 0
    elif mutation == "shape":
        positions = positions.flatten()
    else:
        positions = positions.float()
    state["bert.embeddings.position_ids"] = positions
    torch.save(state, tmp_path / "pytorch_model.bin")
    with pytest.raises(RuntimeError, match="position_ids differs"):
        load_word_head(tmp_path, device="cpu", expected_labels=6,
                       allowed_unused_keys=["bert.embeddings.position_ids"])
