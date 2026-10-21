"""Failure categories, padding oracle identities, and small-word edit counts."""

import json

import pytest

from scripts.analyze_mask_to_crop_failures import at_most_one_edit, diagnose


@pytest.fixture
def cfg():
    return json.loads(open('configs/approximate/mask_to_crop_failure_research_v1.json').read())


def shard(proposals, variants=None):
    return {'sample_id':'sample:1','image_hw':[128,128],
        'variants': variants or {'1':{'proposals':proposals,'candidates_before_cap':len(proposals)}},
        'mask_measurements':{'components_8_connected':[{'bbox_native':p['bbox'],'label':p['component_label']} for p in proposals]}}


def proposal(cid, box, key):
    return {'contour_index':cid,'component_label':cid+1,'bbox_native':box,'bbox':box,'padded_bbox_original':box,'crop_array_sha256':key}


def test_nested_failure_categories_are_exclusive_and_keep_missing_words(cfg):
    p=[proposal(0,[0,0,10,10],'a'),proposal(1,[20,0,30,10],'b')]
    gt=[{'bbox':[0,0,10,10],'text':'A'},{'bbox':[20,0,30,10],'text':'B'},
        {'bbox':[40,0,50,10],'text':'C'}]
    rows,s=diagnose(shard(p),gt,{'a':{'raw_text':'A'},'b':{'raw_text':'X'}},cfg)
    assert [r['category'] for r in rows]==['correct','localized_ocr_or_crop_error','not_localized']
    assert s['exact']==1 and s['localized']==2 and s['gt_words']==3


def test_padding_oracle_keeps_one_word_per_contour_and_fixed_baseline_ids(cfg):
    a=proposal(1,[0,0,10,10],'a');b=proposal(1,[0,0,10,10],'b')
    extra=proposal(2,[20,0,30,10],'c')
    variants={'1':{'proposals':[a],'candidates_before_cap':1},'2':{'proposals':[b,extra],'candidates_before_cap':2}}
    gt=[{'bbox':[0,0,10,10],'text':'A'},{'bbox':[0,0,10,10],'text':'B'},
        {'bbox':[20,0,30,10],'text':'C'}]
    _,s=diagnose(shard([a],variants),gt,{'a':{'raw_text':'A'},'b':{'raw_text':'B'},'c':{'raw_text':'C'}},cfg)
    assert s['exact']==s['padding_oracle']==1


def test_padding_oracle_can_recover_a_wrong_reading_without_adding_a_contour(cfg):
    a=proposal(1,[0,0,10,10],'a');b=proposal(1,[0,0,10,10],'b')
    variants={'1':{'proposals':[a],'candidates_before_cap':1},'2':{'proposals':[b],'candidates_before_cap':1}}
    _,s=diagnose(shard([a],variants),[{'bbox':[0,0,10,10],'text':'YES'}],
                 {'a':{'raw_text':'NO'},'b':{'raw_text':'YES'}},cfg)
    assert s['exact']==0 and s['padding_oracle']==1


def test_detection_box_must_not_change_under_padding_views(cfg):
    a=proposal(1,[0,0,10,10],'a');b=proposal(1,[0,0,11,10],'b')
    variants={'1':{'proposals':[a],'candidates_before_cap':1},'2':{'proposals':[b],'candidates_before_cap':1}}
    with pytest.raises(ValueError,match='detection box changed'):
        diagnose(shard([a],variants),[{'bbox':[0,0,10,10],'text':'A'}],{'a':{'raw_text':'A'},'b':{'raw_text':'A'}},cfg)


def test_empty_predictions_keep_all_words_in_denominator(cfg):
    rows,s=diagnose(shard([]),[{'bbox':[0,0,10,10],'text':'A'}],{},cfg)
    assert s['gt_words']==1 and s['exact']==s['localized']==0 and rows[0]['category']=='not_localized'


def test_below_cap_components_measure_geometry_without_fabricating_ocr(cfg):
    p=proposal(0,[0,0,10,10],'a');source=shard([p])
    source['mask_measurements']['components_8_connected'].append({'bbox_native':[20,0,30,10],'label':2})
    source['variants']['1']['candidates_before_cap']=2
    rows,s=diagnose(source,[{'bbox':[0,0,10,10],'text':'A'},{'bbox':[20,0,30,10],'text':'B'}],{'a':{'raw_text':'A'}},cfg)
    assert s['localized']==1 and s['uncapped_geometry']==2 and s['exact']==1
    assert rows[1]['localized_uncapped_geometry'] and rows[1]['category']=='not_localized'


@pytest.mark.parametrize('a,b,expected',[('word','Word',True),('word','work',True),('word','words',True),
    ('words','word',True),('','a',True),('word','world',True),('word','workx',False),('cat','dog',False)])
def test_one_edit_diagnostic(a,b,expected):
    assert at_most_one_edit(a,b)==expected
