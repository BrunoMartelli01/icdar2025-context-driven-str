import json

import pytest

from recognition.paper_image_router_evaluation import word_metrics,read_word_branches
from recognition.clip_router_evaluation import evaluate_predictions
from scripts.crop_clip_runtime import write_json,sha
from scripts.run_clip_high_recall_admission import payload_digest


def pred(text,**kwargs):return {'text':text,'ignored':False,**kwargs}


def test_t1_is_one_unlocalized_prediction_and_other_gt_words_remain_missed():
    gt=[{'text':'SHOP','bbox':[100,100,200,200]},{'text':'OPEN','bbox':[300,300,400,400]}]
    counts=word_metrics([pred('shop',unlocalized=True)],gt)
    assert (counts['tp'],counts['fp'],counts['fn'],counts['detected_words'])==(1,0,1,1)
    assert counts['matched_gt_indices']==[0] and counts['invalid_envelopes']==0
    # The identical word criterion applies to localized crop and detector outputs.
    assert word_metrics([pred('shop',bbox=[0,0,1,1])],gt)==counts


def test_duplicate_matching_casefold_and_punctuation_are_preserved():
    counts=word_metrics([pred('STRASSE'),pred('ltd!'),pred(''),pred('open',ignored=True)],
        [{'text':'Straße'},{'text':'STRASSE'},{'text':'LTD'}])
    assert (counts['tp'],counts['fp'],counts['fn'])==(1,2,2)
    assert counts['all_outputs']==4 and counts['ignored_outputs']==1


def test_invalid_envelopes_remain_unmatchable_false_positives():
    counts=word_metrics([pred('shop',bbox=None,invalid_envelope={'clipped_bbox':[0,0,0,0]})],
        [{'text':'shop'}])
    assert (counts['tp'],counts['fp'],counts['fn'])==(0,1,1)
    assert counts['invalid_envelopes']==1 and counts['all_outputs']==1


def test_t1_raw_sequence_is_not_split_into_multiple_predictions():
    counts=word_metrics([pred('NEW YORK',unlocalized=True)],[{'text':'NEW'},{'text':'YORK'}])
    assert counts['all_outputs']==1 and counts['tp']==0 and counts['fn']==2


@pytest.mark.parametrize('prediction',[pred(None),{'text':'shop','ignored':1}])
def test_invalid_word_metric_inputs_fail(prediction):
    with pytest.raises(ValueError,match='text/ignore'):word_metrics([prediction],[{'text':'shop'}])


def reader_fixture(tmp_path):
    observed={};frozen={};t1={};reports={};gt=[{'text':'SHOP','bbox':[0,0,10,10]}]
    for split,sid in [('train','a'),('test','b')]:
        sample={'sample_id':sid,'image_sha256':'rgb','image_path':'unused',
            'rows':[{'instance_id':1,'raw_text':'SHOP','bbox':[0,0,10,10]}]}
        observed[split]=[sample];route={'observed':sample,'ignored_ids':[],
            'contract':{'plan_sha256':'plan','sample_id':sid,'image_sha256':'rgb'}}
        route['payload_sha256']=payload_digest(route)
        write_json(tmp_path/split/'routes'/(sid+'.json'),route)
        predictions=[pred('SHOP',bbox=[0,0,10,10])]
        metric=evaluate_predictions(predictions,gt,iou_threshold=.5)
        result={'sample_id':sid,'ground_truth':gt,'predictions_all':predictions,
            'action':'deepsolo' if split=='test' else 'accept_parseq'}
        reports[split]={'status':'SUCCESS','plan_sha256':'plan','per_image':[result]}
        write_json(tmp_path/split/'results.json',reports[split])
        frozen[split]=[{'sample_id':sid,'parseq':metric,'deepsolo':metric if split=='test' else None}]
        t1[sid]={'result':{'raw_text':'shop'}}
    audit={'status':'PASS','plan_sha256':'plan','result_sha256':sha(tmp_path/'test/results.json'),
        'deepsolo_only_per_image':[{'sample_id':'b','predictions':[pred('SHOP',bbox=[0,0,10,10])]}]}
    write_json(tmp_path/'test_independent_audit.json',audit)
    cfg={'source_plan_sha256':'plan','geometry_matching_iou':.5,'word_evaluation_sources':{
        n:sha(tmp_path/n) for n in ['train/results.json','test/results.json','test_independent_audit.json']}}
    return cfg,observed,frozen,t1


def test_reader_reuses_frozen_predictions_gt_and_preserves_missing_fallback(tmp_path):
    cfg,observed,frozen,t1=reader_fixture(tmp_path)
    report=read_word_branches(cfg,tmp_path,observed,frozen,t1)
    train=report['splits']['train'][0];test=report['splits']['test'][0]
    assert train['t1']['detected_words']==1 and train['t1']['tp']==1
    assert train['deepsolo'] is None and test['deepsolo']['tp']==1
    assert report['ground_truth_in_routing'] is False
    assert len(report['input_sha256'])==5


@pytest.mark.parametrize('fault',['source_hash','audit_population','geometry_counts','route_payload'])
def test_reader_refuses_changed_sources_populations_geometry_or_payload(tmp_path,fault):
    cfg,observed,frozen,t1=reader_fixture(tmp_path)
    if fault=='source_hash':(tmp_path/'train/results.json').write_text('{}')
    elif fault=='audit_population':
        path=tmp_path/'test_independent_audit.json';audit=json.loads(path.read_text())
        audit['deepsolo_only_per_image']*=2;write_json(path,audit)
        cfg['word_evaluation_sources']['test_independent_audit.json']=sha(path)
    elif fault=='geometry_counts':frozen['train'][0]['parseq']['tp']=0
    else:
        path=tmp_path/'train/routes/a.json';route=json.loads(path.read_text())
        route['ignored_ids']=[1];write_json(path,route)
    with pytest.raises(ValueError):read_word_branches(cfg,tmp_path,observed,frozen,t1)
