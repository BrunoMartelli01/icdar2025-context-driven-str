import numpy as np
import pytest
import torch

from recognition.clip_router_prompt_ablation import (mutations, prompt_chunks,
    transcript, fit_reference, image_scores, coverage_threshold, budget_accept)
from scripts.run_clip_router_prompt_ablation import Encoder, evaluate, model_identity


class CharTokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': [0, *range(len(text)), 1]}


def test_chunking_preserves_every_character_and_prefix_context():
    text='hello supercalifragilistic TEST'
    chunks=prompt_chunks('text: {text}',text,CharTokenizer(),17)
    assert ''.join(s.removeprefix('text: ') for s in chunks)==text
    assert all(len(CharTokenizer()(s)['input_ids'])<=17 for s in chunks)
    assert len(chunks)>1
    with pytest.raises(ValueError,match='prefix'):
        prompt_chunks('too many: {text}','a',CharTokenizer(),5)


@pytest.mark.parametrize('text',['x','A','hello','aa','café','123',' X '])
def test_spelling_controls_deterministic_nonempty_and_not_original(text):
    controls=mutations(text)
    assert controls and controls==mutations(text)
    assert text not in controls and all(s.strip() for s in controls)


def test_transcript_reads_all_outputs_in_geometry_order():
    rows=[{'instance_id':1,'bbox':[0,10,4,15],'raw_text':'B'},
          {'instance_id':2,'bbox':[10,0,14,5],'raw_text':'A'}]
    assert transcript(rows)=='A B'


def feature():
    return {'eligible':True,'word_feature_names':['baseline','margin'],
        'word_features':[{'baseline':.1,'margin':-.01},{'baseline':.3,'margin':.02}],
        'native':[.9,.8],'scene_features':{'all_en':.2}}


def test_calibration_only_and_weakest_word_and_blank_force_fallback():
    fs={'cal':feature(),'test':feature()}
    ref=fit_reference(fs,['cal'])
    fs['test']['word_features'][0]['baseline']=10
    assert ref['word']['baseline']==[.1,.3]
    f=feature();scores=image_scores(f,ref)
    assert scores['crop_baseline']==pytest.approx((.75*.25)**.5)
    f['eligible']=False
    assert all(v is None for v in image_scores(f,ref).values())


def test_coverage_ties_inclusive_and_budget_ties_deterministic():
    scores={'a':.1,'b':.5,'c':.5,'d':None}
    assert coverage_threshold(scores,list(scores),.5)==.5
    assert budget_accept(scores,list(scores),1)=={'b'}
    assert budget_accept(scores,list(scores),4)=={'a','b','c'}
    assert coverage_threshold({'a':None},['a'],.5) is None


def test_missing_fallback_invalidates_full_metrics_and_counts_kept_words():
    m={'tp':1,'fp':0,'fn':1,'tn':0,'detected_words':1,'gt_words':2,
       'all_outputs':1,'ignored_outputs':0,'invalid_envelopes':0,'matched_gt_indices':[0]}
    rows=[{'sample_id':'a','score':0,'parseq':m,'deepsolo':None}]
    p=evaluate(rows,{'a':.5},.5,['a'],2)
    assert p['words_retained_parseq']==1 and p['safe_images']==1 and p['complete_images']==0
    p=evaluate(rows,{'a':None},.5,['a'],2)
    assert p['combined_model'] is None and p['missing_deepsolo_sample_ids']==['a']


def test_siglip_fixed_padding_no_attention_mask_and_normalized_features():
    calls=[]
    class Tokenizer:
        def __call__(self, text, **kwargs):
            calls.append(kwargs)
            return {'input_ids':torch.ones((len(text),8),dtype=torch.long),'attention_mask':torch.ones((len(text),8))} if isinstance(text,list) else {'input_ids':[1,2]}
    class Model:
        def get_text_features(self, **kwargs):
            assert set(kwargs)=={'input_ids'}
            return torch.tensor([[3.,4.]]).repeat(len(kwargs['input_ids']),1)
    e=object.__new__(Encoder);e.tokenizer=Tokenizer();e.model=Model();e.limit=8
    e.backend='siglip';e.device='cpu';e.batch_size=2
    result=e.embeddings(['hello','word'])
    assert result.shape==(2,2)
    np.testing.assert_allclose(result,[[.6,.8],[.6,.8]],atol=1e-7)
    assert float((result[0]**2).sum())==pytest.approx(1.)
    assert calls[-1]['padding']=='max_length' and calls[-1]['max_length']==8
    assert calls[-1]['truncation'] is False


def test_model_refuses_unpinned_revision(tmp_path):
    with pytest.raises(ValueError,match='revision'):
        model_identity({'revision':'main'},tmp_path)


def test_siglip_real_class_api_without_download():
    """Structural compatibility only: random tiny model, not checkpoint qualification."""
    from transformers import SiglipConfig, SiglipModel
    config=SiglipConfig(text_config={'vocab_size':16,'hidden_size':16,
        'intermediate_size':32,'num_attention_heads':4,'num_hidden_layers':1,'max_position_embeddings':8},
        vision_config={'hidden_size':16,'intermediate_size':32,'num_attention_heads':4,
            'num_hidden_layers':1,'image_size':8,'patch_size':2})
    model=SiglipModel(config).eval()
    with torch.inference_mode():
        text=model.get_text_features(input_ids=torch.ones((2,8),dtype=torch.long))
        image=model.get_image_features(pixel_values=torch.zeros((2,3,8,8)))
    assert text.shape==image.shape==(2,16)
    assert torch.isfinite(text).all() and torch.isfinite(image).all()


def test_model_summary_chooses_by_train_despite_opposite_test_metrics(tmp_path):
    from scripts.crop_clip_runtime import sha, write_json
    from scripts.summarize_clip_router_prompt_ablation import summary
    paths=tmp_path/'paths.json';write_json(paths,{'output_root':str(tmp_path)})
    for name,validation_precision,test_precision in [('a',.9,.1),('b',.8,.99)]:
        root=tmp_path/name/'run'
        plan={'config':{},'source_sha256':{},'branch_metrics_sha256':'same','code_sha256':{}}
        write_json(root/'resolved_configuration.json',plan)
        chosen={'winner':'f','families':{'f':{'validation':{'precision_retained_parseq':validation_precision,'words_retained_parseq':10}}}}
        write_json(root/'train_selection.json',chosen)
        point={'threshold':.5,'parseq_images':1,'words_retained_parseq':10,
            'correct_words_retained_parseq':1,'precision_retained_parseq':test_precision,
            'safe_images':0,'complete_images':0,'missing_deepsolo_sample_ids':[],'combined_model':None}
        result={'model':name,'status':'SUCCESS','result_class':'approximate reproduction',
            'reproduction_mode':'approximate','selection_uses_test_labels':False,
            'train_selection_sha256':sha(root/'train_selection.json'),
            'plan_sha256':sha(root/'resolved_configuration.json'),
            'comparisons':{'f':{'test':{'same_budget_descriptive':point}}}}
        write_json(root/'results.json',result)
        write_json(tmp_path/name/'latest.json',{'results_path':str(root/'results.json'),
            'results_sha256':sha(root/'results.json')})
    result=summary(paths,['a','b'],tmp_path/'summary')
    assert result['winner_train']['model']=='a'
    assert result['selection_uses_test_labels'] is False
