import copy

import pytest

from recognition.clip_router_threshold_sweep import curves, point, validate_population


def metrics(tp,fp,fn):
    return dict(tp=tp,fp=fp,fn=fn,tn=0,detected_words=tp+fp,gt_words=tp+fn,
                all_outputs=tp+fp,ignored_outputs=0,invalid_envelopes=0)


def samples():
    return [{'sample_id':'a','score':.2,'parseq':metrics(1,2,1),'deepsolo':metrics(2,0,0)},
            {'sample_id':'b','score':.8,'parseq':metrics(1,0,0),'deepsolo':metrics(0,3,1)}]


def test_threshold_switches_whole_image_and_word_count_can_increase():
    rows=samples();result=curves(rows,[0,.2,.5,1],['a','b'],3)
    a,b,c,d=result['grid']
    assert a['words_retained_parseq']==4 and a['combined_model']['detected_words']==4
    assert b['parseq_images']==2  # Inclusive threshold.
    assert c['parseq_images']==1 and c['words_retained_parseq']==1
    assert c['words_from_deepsolo']==2 and c['combined_model']['detected_words']==3
    assert c['combined_model']['accuracy_tn0']==1
    # Rejecting the next image swaps its single word for three detector outputs.
    assert d['combined_model']['detected_words']==5 and d['parseq_images']==0
    assert d['combined_model']['fp']==3 and d['combined_model']['fn']==1
    assert d['combined_model']['accuracy_tn0']==pytest.approx(2/6)


def test_missing_fallback_never_becomes_empty_or_omitted():
    rows=samples();rows[0]['deepsolo']=None
    result=curves(rows,[.1,.5],['a','b'],3)
    assert result['grid'][0]['status']=='COMPLETE'
    incomplete=result['grid'][1]
    assert incomplete['status']=='MISSING_DEEPSOLO_OUTPUTS'
    assert incomplete['combined_model'] is None and incomplete['words_from_deepsolo'] is None
    assert incomplete['images']==2 and incomplete['missing_deepsolo_sample_ids']==['a']


def test_score_none_forces_fallback_even_at_zero_and_curve_is_readonly():
    rows=samples();rows[0]['score']=None;before=copy.deepcopy(rows)
    result=curves(rows,[0,.5,1],['a','b'],3)
    assert result['grid'][0]['parseq_images']==1
    assert result['threshold_selected_on_test'] is False
    assert rows==before


@pytest.mark.parametrize('change',[
    lambda rows:rows.pop(),
    lambda rows:rows.append(copy.deepcopy(rows[0])),
    lambda rows:rows[0]['parseq'].update(gt_words=99),
    lambda rows:rows[0]['parseq'].update(fp=-1),
    lambda rows:rows[0]['deepsolo'].update(fn=3,gt_words=5),
    lambda rows:rows[0].update(parseq=None),
])
def test_population_and_denominator_guards(change):
    rows=samples();change(rows)
    with pytest.raises((ValueError,TypeError)):validate_population(rows,['a','b'],3)


def test_empty_population_fails_actionably():
    with pytest.raises(ValueError):validate_population([],[],0)


def test_csv_and_offline_visuals_preserve_missing_points(tmp_path):
    import csv
    import xml.etree.ElementTree as ET
    from scripts.clip_router_sweep_visuals import render
    from scripts.sweep_clip_router_e2e import write_csv
    rows=samples();train=copy.deepcopy(rows);train[0]['deepsolo']=None
    report={'splits':{'test':curves(rows,[0,.5,1],['a','b'],3),
                      'train':curves(train,[0,.5,1],['a','b'],3)}}
    write_csv(tmp_path/'thresholds.csv',report);render(tmp_path,report,.5)
    table=list(csv.DictReader((tmp_path/'thresholds.csv').open()))
    missing=next(r for r in table if r['split']=='train' and r['threshold']=='0.5')
    assert missing['status']=='MISSING_DEEPSOLO_OUTPUTS' and missing['accuracy_tn0']==''
    assert missing['missing_deepsolo_images']=='1'
    assert ET.parse(tmp_path/'threshold_curve.svg').getroot().tag.endswith('svg')
    page=(tmp_path/'explore.html').read_text()
    assert 'application/json' in page and 'DATA_JSON' not in page and 'REFERENCE_VALUE' not in page


def test_cached_only_stage_never_executes_missing_gpu_predictions(tmp_path,monkeypatch):
    from scripts import sweep_clip_router_e2e as script
    rows=samples();rows[0]['deepsolo']=None
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected GPU inference')
    monkeypatch.setattr(script,'DeepSoloSubprocessAdapter',forbidden)
    contexts={'a':{'observed':{'image_sha256':'image'},'result':object()}}
    before=copy.deepcopy(rows)
    assert script.complete_train(rows,contexts,{'fallback_config':{},'fallback_paths':{}},
        tmp_path,'job','source',False)=={}
    assert rows==before


def test_additional_train_cache_binds_to_source_plan(tmp_path):
    from scripts.sweep_clip_router_e2e import complete_train
    from scripts.run_clip_router_e2e import save_shard
    rows=samples();rows[0]['deepsolo']=None
    save_shard(tmp_path/'additional_train_deepsolo/a.json',
        {'plan_sha256':'job','source_plan_sha256':'wrong','sample_id':'a','image_sha256':'image'},
        {'response':{}})
    with pytest.raises(ValueError,match='contract'):
        complete_train(rows,{'a':{'observed':{'image_sha256':'image'}}},
            {'fallback_config':{},'fallback_paths':{}},tmp_path,'job','source',False)


def test_missing_gpu_fails_before_creating_a_prediction(tmp_path,monkeypatch):
    from scripts import sweep_clip_router_e2e as script
    rows=samples();rows[0]['deepsolo']=None
    monkeypatch.setattr(script,'sha',lambda path:'hash')
    monkeypatch.setattr(script,'verify_host_source_identity',lambda *args:
        {'candidates':{'candidate':{'checkpoint_sha256':'hash','checkpoint_path':'unused'}}})
    monkeypatch.setattr(script.torch.cuda,'is_available',lambda:False)
    cfg={'candidate':'candidate','runtime_sha256':'hash','checkpoint_sha256':'hash'}
    paths={'deepsolo_runtime_manifest':'unused','deepsolo_source_root':'unused'}
    with pytest.raises(RuntimeError,match='CUDA unavailable'):
        script.complete_train(rows,{'a':{'observed':{'image_sha256':'image'}}},
            {'fallback_config':cfg,'fallback_paths':paths},tmp_path,'job','source',True)
    assert not (tmp_path/'additional_train_deepsolo/a.json').exists()
