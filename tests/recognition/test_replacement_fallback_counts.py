import hashlib
import json
import pytest
from recognition.replacement_fallback_counts import replacement_count,load_counts


def fixture(tmp_path):
    record=dict(sample_id="icdar13:train:1",image_sha256="rgb")
    data=dict(mode="approximate",claim_label="approximate reproduction",samples={record['sample_id']:dict(image_sha256="rgb",counts=dict(tp=1))})
    p=tmp_path/'counts.json';p.write_text(json.dumps(data))
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    config=dict(replacement_fallback_counts=dict(icdar13=dict(path=str(p),sha256=digest)))
    return config,record,p


def test_replacement_is_exact_identity_bound_and_returns_copy(tmp_path):
    c,r,p=fixture(tmp_path)
    result=replacement_count(c,r);result['tp']=9
    assert replacement_count(c,r)==dict(tp=1)
    with pytest.raises(ValueError):replacement_count(c,dict(r,image_sha256="other"))


def test_changed_archive_rejected_before_loading(tmp_path):
    c,r,p=fixture(tmp_path);p.write_text('{}')
    load_counts.cache_clear()
    with pytest.raises(ValueError):replacement_count(c,r)


def test_train_weak_pair_serialization_preserves_literal_spaces(tmp_path):
    from recognition.deepsolo_lexicon import DeepSoloLexicon
    path=tmp_path/'pairs.json'
    canonical={"NEW YORK":"New York", "A&B":"A&B"}
    path.write_text(json.dumps(canonical))
    model=DeepSoloLexicon(sorted(canonical),json.loads(path.read_text()),distance_exclusive=1.5)
    selected,_=model.correct([dict(instance_id=0,text="NEW YORX"),dict(instance_id=1,text="A&B")],unmatched_policy="source_reject")
    assert [p['text'] for p in selected]==["New York","A&B"]
