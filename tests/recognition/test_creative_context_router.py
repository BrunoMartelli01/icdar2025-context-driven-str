"""Scientific boundaries for six context signals and TRAIN error retrieval."""
import json
from pathlib import Path

import numpy as np
import pytest

from recognition.creative_context_signals import (
    ContextRenderer,context_views,expanded_job,glyph_features,permutation_features,
    structure_features,swap_features,
)
from recognition.context_memory_router import fit_memory,predict_memory

ROOT=Path(__file__).resolve().parents[2]
CFG=json.loads((ROOT/'configs/approximate/creative_context_router_v1.json').read_text())
PATHS=json.loads((ROOT/'configs/approximate/creative_context_router_local_paths_v1.json').read_text())


def row(cid,box,text,confidence=.99):
    return {'candidate_id':cid,'bbox':box,'raw_text':text,'confidence':confidence}


def test_transplant_preserves_word_guard_pixels_location_and_canvas():
    rgb=np.arange(40*60*3,dtype=np.uint16).reshape(40,60,3).astype(np.uint8)
    donor=np.zeros_like(rgb);job={'target_box':[20,15,30,25],'context_box':[15,10,35,30]}
    job=expanded_job(job,[60,40],CFG['swap_settings']);views=context_views(rgb,job,donor,CFG['swap_settings'])
    x,y,_,_=job['context_box'];a,b,c,d=job['protected_box']
    for name in ('original','neutral','foreign'):
        assert np.array_equal(views[name][b-y:d-y,a-x:c-x],rgb[b:d,a:c])
        assert views[name].shape==views['original'].shape
    assert not np.array_equal(views['foreign'],views['foreign_without_target'])


def test_invalid_guard_is_an_error():
    with pytest.raises(ValueError,match='Protected'):
        context_views(np.zeros((10,10,3),np.uint8),{'target_box':[2,2,3,3],'context_box':[0,0,5,5],'protected_box':[-1,0,3,3]},np.zeros((10,10,3),np.uint8),CFG['swap_settings'])


def test_context_stability_does_not_certify_visual_support():
    answers={k:[{'yes_probability':.9,'yes_logprob':-1.,'no_logprob':-3.},{'yes_probability':.8}]
             for k in ('original','neutral','foreign','foreign_without_target')}
    f=swap_features([{'answers':answers}])
    assert f['swap_stability_mean']==1 and f['swap_foreign_evidence_mean']==0


def test_permutation_character_summary_excludes_eos():
    words=[{'text':'AB','target_probabilities':[[.9,.8,0],[.9,.8,1]],'argmax_ids':[[1,2,0],[1,2,4]]}]
    f=permutation_features(words)
    assert f['permutation_agreement_mean']==1 and f['permutation_variation_mean']==0
    assert np.isclose(f['permutation_support_mean'],.85)


def test_repeated_glyph_conflict_requires_distinct_neighboring_words():
    rows=[row('a',[0,0,10,10],'A'),row('b',[12,0,22,10],'B')]
    words=[{'candidate_id':r['candidate_id'],'characters':[{'position':0,'character':r['raw_text'],'probability':.99,'vector':[1.,0.]}]} for r in rows]
    f,e=glyph_features(words,rows,CFG['glyph_settings'],CFG['geometry'])
    assert len(e['pairs'])==1 and f['glyph_contradiction']==1
    f,_=glyph_features(words[:1],rows[:1],CFG['glyph_settings'],CFG['geometry'])
    assert f['glyph_missing']==1 and f['glyph_contradiction']==0


def test_duplicate_overlapping_word_boxes_are_not_independent_glyph_evidence():
    rows=[row('a',[0,0,10,10],'A'),row('b',[0,0,10,10],'B')]
    words=[{'candidate_id':r['candidate_id'],'characters':[{'position':0,'character':r['raw_text'],'probability':.99,'vector':[1.,0.]}]} for r in rows]
    f,e=glyph_features(words,rows,CFG['glyph_settings'],CFG['geometry'])
    assert not e['pairs'] and f['glyph_missing']==1


def test_price_structure_abstains_without_menu_context():
    rows=[row('a',[0,0,20,10],'$5.00')]
    f,_=structure_features(rows,[],CFG['geometry']);assert f['structure_missing']==1 and f['structure_orphan_price']==0
    f,_=structure_features(rows,['a menu'],CFG['geometry']);assert f['structure_orphan_price']==1


def test_invalid_time_flag_requires_observed_hours_context():
    rows=[row('a',[0,0,10,10],'25:99')]
    f,_=structure_features(rows,[],CFG['geometry']);assert f['structure_invalid_time']==0
    rows.append(row('b',[30,0,50,10],'OPEN'))
    f,_=structure_features(rows,[],CFG['geometry']);assert f['structure_invalid_time']>0


def test_observed_times_roundtrip_through_immutable_json_cache():
    rows=[row('a',[0,0,30,10],'OPEN'),row('b',[32,0,70,10],'9:00')]
    features,records=structure_features(rows,[],CFG['geometry'])
    assert json.loads(json.dumps(records))==records
    assert json.loads(json.dumps(features))==features


def test_renderer_style_does_not_use_target_as_neighbor():
    from PIL import Image,ImageDraw,ImageFont
    im=Image.new('RGB',(240,40),'white');draw=ImageDraw.Draw(im);font=ImageFont.truetype(PATHS['fonts'][0],25)
    draw.text((0,0),'OPEN',font=font,fill='black');draw.text((115,0),'NOW',font=font,fill='black')
    rows=[row('a',[0,0,100,35],'OPEN'),row('b',[110,0,220,35],'NOW')]
    renderer=ContextRenderer(PATHS['fonts'][:2],CFG['render_settings']);f,e=renderer.infer(np.asarray(im),rows,CFG['geometry'])
    assert len(e)==2
    assert all(r['candidate_id'] not in r.get('neighbor_ids',[]) for r in e)
    f,_=renderer.infer(np.asarray(im),rows[:1],CFG['geometry']);assert f['render_margin_missing']==1
    assert renderer.rendered('日本語',0) is None


def memory_data():
    features={s:{'visual_embedding':v,'values':{'q':i},'image_sha256':rgb} for i,(s,v,rgb) in enumerate([
        ('a',[1.,0.],'same'),('b',[1.,0.],'same'),('c',[0.,1.],'other'),('test',[1.,0.],'query')])}
    branches={s:{'light':{'tp':i,'detected_words':2},'deepsolo':{'tp':0,'detected_words':1}} for i,s in enumerate(['a','b','c'])}
    return features,branches


def test_memory_excludes_same_id_and_same_rgb_even_if_perfect_match():
    features,branches=memory_data();m=fit_memory(features,branches,['a','b','c'],'visual',1,['q'])
    p=predict_memory(m,features,['a'])
    assert p[0,0]==2 # c is the sole different-content reference; b is forbidden.


def test_memory_query_uses_only_train_outcomes():
    features,branches=memory_data();m=fit_memory(features,branches,['a','b','c'],'joint',2,['q'])
    first=predict_memory(m,features,['test'])
    features['test']['ground_truth']='SHOULD_NOT_BE_READ'
    assert np.array_equal(first,predict_memory(m,features,['test']))
    with pytest.raises(ValueError,match='exactly'):
        fit_memory(features,branches,['a','b'],'joint',2,['q'])


def test_capacity_edge_prediction_is_preserved_with_explicit_probe_abstention():
    from recognition.creative_context_runtime import FrozenPermutationProbe
    from types import SimpleNamespace
    probe=FrozenPermutationProbe.__new__(FrozenPermutationProbe);probe.model=SimpleNamespace(max_label_length=25)
    word='A'*26
    out=probe.word(None,{'candidate_id':'long','raw_text':word})
    assert out['text']==word and out['abstention']=='prediction_exceeds_conditional_probe_capacity'
