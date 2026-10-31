import copy
import pytest

from recognition.icdar13_hypothesis_tests import crossfit_folds, vocabulary_rows, independent_counts, gt_crop_diagnosis, raw_evidence_features


def test_folds_keep_duplicate_rgb_together_and_exclude_holdout():
    records = [dict(sample_id=str(i), rgb_sha256=str(i//2)) for i in range(14)]
    folds = crossfit_folds(records, 5, 19)
    assert folds == crossfit_folds(list(reversed(records)), 5, 19)
    assert sorted(s for f in folds for s in f) == sorted(r['sample_id'] for r in records)
    owner = {s:i for i,f in enumerate(folds) for s in f}
    assert all(owner[str(i)] == owner[str(i+1)] for i in range(0,14,2))
    for f in folds:
        fit = {r['sample_id'] for r in records} - set(f)
        assert not fit & set(f)
    with pytest.raises(ValueError): crossfit_folds(records, 8, 19)


def test_lexicon_is_prediction_only_and_protects_geometry_evidence():
    class Lexicon:
        def nearest(self, text): return dict(text='HELLO')
    rows = [dict(raw_text='Hellu', bbox=[1,2,8,9], sequence_confidence_native=.95),
            dict(raw_text='A1', bbox=[2,3,7,8], sequence_confidence_native=.5)]
    before = copy.deepcopy(rows)
    corrected = vocabulary_rows(rows, Lexicon(), 'generic_conservative', dict(minimum_length=4,protect_probability=.99))
    assert corrected[0]['raw_text'] == 'HELLO' and corrected[1]['raw_text'] == 'A1'
    assert corrected[0]['bbox'] == rows[0]['bbox']
    assert corrected[0]['sequence_confidence_native'] == .95 and rows == before
    assert vocabulary_rows(rows, Lexicon(), 'none')[0]['raw_text'] == 'Hellu'


def test_counts_preserve_empty_gt_and_duplicates_and_ignore():
    pred=[dict(text='Hello!', ignored=False),dict(text='HELLO',ignored=False),dict(text='!',ignored=False),dict(text='x',ignored=True)]
    result=independent_counts(pred,[dict(text='hello'),dict(text='!!!')])
    assert (result['tp'],result['fp'],result['fn'],result['gt_words']) == (1,2,1,2)


def test_crop_diagnosis_separates_filter_localization_and_recognizer():
    gt=[dict(text='ONE',bbox=[0,0,10,10]),dict(text='TWO',bbox=[20,0,30,10]),dict(text='THREE',bbox=[40,0,50,10])]
    raw=[dict(raw_text='one',bbox=[0,0,10,10]),dict(raw_text='threa',bbox=[40,0,50,10])]
    result=gt_crop_diagnosis(gt,raw,[],[dict(raw_text=g['text']) for g in gt],.5)
    assert [r['group'] for r in result] == ['correct_crop_rejected_by_filter','no_viable_crop','viable_crop_wrong_reading']
    assert all(r['gt_crop_correct'] for r in result)


def test_raw_descriptors_retain_rejected_evidence_and_cover_empty_emissions():
    raw=[dict(candidate_id='a',raw_text='READ',detector_score=.8,sequence_confidence_native=.7)]
    f=raw_evidence_features(raw,[])
    assert f.shape==(8,) and f[-1]==1 and f[4]==.8 and f[6]==4
    assert raw_evidence_features([],[])[-1]==1


def test_training_entry_rejects_strict_before_loading_any_artifact():
    from types import SimpleNamespace
    from pathlib import Path
    from scripts.run_icdar13_hypothesis_tests import prepare
    with pytest.raises(ValueError,match='APPROXIMATE'):
        prepare(SimpleNamespace(mode='strict',config=Path('configs/strict/missing.json')))


def test_training_rejects_changed_initial_weights_before_gpu(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import run_icdar13_hypothesis_tests as runner
    monkeypatch.setattr(runner,'REPO',tmp_path)
    config=tmp_path/'configs/approximate/test.json';config.parent.mkdir(parents=True)
    light=tmp_path/'light';light.mkdir();weights=tmp_path/'weights.pt';weights.write_bytes(b'changed bytes')
    (light/'resolved_configuration.json').write_text(json.dumps(dict(paths=dict(tt_checkpoint=str(weights),tt_checkpoint_sha256='0'*64))))
    config.write_text(json.dumps(dict(mode='approximate',pinned_inputs={},light_root=str(light))))
    with pytest.raises(ValueError,match='initialization artifact'):
        runner.prepare(SimpleNamespace(mode='approximate',config=config))


def test_independent_csv_recount_exposes_full_aggregate_schema(tmp_path,monkeypatch):
    from scripts import report_icdar13_hypothesis_tests as report
    from scripts.icdar_budget_data import sha
    annotation=tmp_path/'gt.csv';annotation.write_text('0,0,9,9,Hello\n')
    record=dict(annotation_path=str(annotation),annotation_sha256=sha(annotation))
    monkeypatch.setattr(report,'annotation',lambda record:([],None))
    monkeypatch.setattr(report,'predictions',lambda rows,record,valid:[
        dict(text='hello',ignored=False),dict(text='hidden',ignored=True)])
    result=report.csv_recount([],record)
    assert (result['tp'],result['fp'],result['fn'])==(1,0,0)
    assert (result['all_outputs'],result['ignored_outputs'])==(2,1)
    assert report.aggregate([result])['f1']==1.
