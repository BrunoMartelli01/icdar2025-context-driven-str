"""Scientific boundary checks for v3; tiny random models are structural tests."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from transformers import CLIPConfig, CLIPModel

from recognition.clip_literature_runtime import (
    LiteratureEncoder, LowercaseTokenizer, PublishedSquashProcessor,
    validate_native_architecture,
)
from recognition.clip_router_prompt_ablation import prompt_chunks
from scripts import run_clip_router_literature_v3 as runner


def test_siglip_lowercase_used_for_both_length_and_embedding_inputs():
    calls=[]
    class Tokenizer:
        def __call__(self,text,**kwargs):
            calls.append((text,kwargs))
            return {'input_ids':list(range(len(text)+1))}
    tokenizer=LowercaseTokenizer(Tokenizer())
    text='AİBCDEFG'
    chunks=prompt_chunks('{text}',text,tokenizer,5)
    assert ''.join(chunks)==text
    assert all(len(tokenizer(c)['input_ids'])<=5 for c in chunks)
    tokenizer(['TEXT','İ'],truncation=False)
    assert calls[-1]==(['text','i\u0307'],{'truncation':False})
    with pytest.raises(ValueError,match='raw strings'):
        tokenizer([[1,2]])


def native_fixture():
    return {'model_cfg':{'embed_dim':8,'vision_cfg':{'image_size':8,'width':16,
        'layers':1,'patch_size':2,'head_width':4},'text_cfg':{'context_length':8,
        'vocab_size':16,'width':16,'heads':4,'layers':1}},
        'preprocess_cfg':{'resize_mode':'squash','interpolation':'bicubic',
            'mean':[.1,.2,.3],'std':[.5,.25,.2]}}


def config_fixture(extra=False):
    return CLIPConfig(projection_dim=8,text_config={'vocab_size':17 if extra else 16,
        'hidden_size':16,'intermediate_size':64,'num_attention_heads':4,
        'num_hidden_layers':1,'max_position_embeddings':8,'hidden_act':'gelu',
        'bos_token_id':0,'eos_token_id':15 if extra else 2,'pad_token_id':16 if extra else 1},
        vision_config={'hidden_size':16,'intermediate_size':64,'num_attention_heads':4,
            'num_hidden_layers':1,'image_size':8,'patch_size':2,'hidden_act':'gelu'})


class PublisherTokenizer:
    bos_token_id=14;eos_token_id=15;pad_token_id=15
    def __len__(self):return 16
    def __call__(self,text,**kwargs):
        return {'input_ids':torch.tensor([[14,1,15,15],[14,1,2,15]]),
                'attention_mask':torch.tensor([[1,1,1,0],[1,1,1,1]])}


def spec_fixture(extra=False):
    return {'native_vision_head_width':4,'native_vocab_policy':
            'publisher_extra_unused_pad_row' if extra else 'identical'}


def test_native_processor_matches_published_resize_and_normalization():
    rgb=np.arange(4*12*3,dtype=np.uint8).reshape(4,12,3)
    image=Image.fromarray(rgb)
    processor=PublishedSquashProcessor(native_fixture(),8)
    got=processor(images=[image],return_tensors='pt',input_data_format='channels_last')['pixel_values']
    resized=np.asarray(image.resize((8,8),Image.Resampling.BICUBIC),dtype=np.float32)/255
    expected=(resized-np.array([.1,.2,.3],np.float32))/np.array([.5,.25,.2],np.float32)
    assert got.shape==(1,3,8,8) and got.dtype==torch.float32
    np.testing.assert_allclose(got[0].permute(1,2,0),expected,atol=2e-7)
    with pytest.raises(ValueError,match='PIL RGB'):
        processor(images=[image.convert('L')],return_tensors='pt',input_data_format='channels_last')


@pytest.mark.parametrize('key,value',[('resize_mode','shortest'),('interpolation','bilinear'),
    ('mean',[0,0,float('nan')]),('std',[1,0,1])])
def test_native_processor_refuses_guessed_or_invalid_profile(key,value):
    native=native_fixture();native['preprocess_cfg'][key]=value
    with pytest.raises(ValueError,match='Unsupported published processor'):
        PublishedSquashProcessor(native,8)


@pytest.mark.parametrize('extra',[False,True])
def test_published_native_and_hf_conversion_contract(extra):
    validate_native_architecture(native_fixture(),config_fixture(extra),
                                 spec_fixture(extra),PublisherTokenizer())


@pytest.mark.parametrize('fault,match',[('activation','activation'),('heads','attention heads'),
    ('vocab','vocabulary'),('extra_without_policy','vocabulary'),('tokenizer_pad','special IDs'),
    ('pool','different EOS'),('layers','dimensions')])
def test_scientifically_material_conversion_mismatches_fail(fault,match):
    config=config_fixture();spec=spec_fixture();tokenizer=PublisherTokenizer()
    if fault=='activation':config.vision_config.hidden_act='quick_gelu'
    if fault=='heads':config.vision_config.num_attention_heads=2
    if fault=='vocab':config.text_config.vocab_size+=2
    if fault=='extra_without_policy':config=config_fixture(True)
    if fault=='tokenizer_pad':tokenizer.pad_token_id=0
    if fault=='pool':config.text_config.eos_token_id=3
    if fault=='layers':config.text_config.num_hidden_layers+=1
    with pytest.raises(ValueError,match=match):
        validate_native_architecture(native_fixture(),config,spec,tokenizer)


@pytest.mark.parametrize('extra',[False,True])
def test_real_hf_clip_pools_first_eos_and_ignores_future_pad_rows(extra):
    """Random tiny weights prove API/pooling behavior, not pretrained equivalence."""
    torch.manual_seed(12)
    model=CLIPModel(config_fixture(extra)).eval()
    pad=16 if extra else 15
    ids=torch.tensor([[14,1,15,pad,pad]])
    mask=torch.tensor([[1,1,1,0,0]])
    with torch.inference_mode():
        output=model.text_model(input_ids=ids,attention_mask=mask)
        short=model.get_text_features(input_ids=ids[:,:3],attention_mask=mask[:,:3])
        full=model.get_text_features(input_ids=ids,attention_mask=mask)
        torch.testing.assert_close(output.pooler_output,output.last_hidden_state[:,2])
        torch.testing.assert_close(short,full,atol=1e-6,rtol=1e-6)
        if extra:
            model.text_model.embeddings.token_embedding.weight[pad].fill_(1000)
            changed=model.get_text_features(input_ids=ids,attention_mask=mask)
            torch.testing.assert_close(full,changed,atol=1e-6,rtol=1e-6)


def weight_spec():
    return {'model_id':'publisher/model','revision':'a'*40,'weight_file':'pytorch_model.bin',
            'weight_sha256':None,'backend':'clip','image_size':8}


def hub_info(spec,digest):
    return SimpleNamespace(sha=spec['revision'],siblings=[SimpleNamespace(
        rfilename=spec['weight_file'],lfs=SimpleNamespace(sha256=digest))])


def test_download_resolves_published_checksum_before_bytes_and_seals_identity(tmp_path,monkeypatch):
    spec=weight_spec();content=b'pinned weights';digest=hashlib.sha256(content).hexdigest();events=[]
    class Hub:
        def model_info(self,model_id,**kwargs):
            events.append('metadata')
            assert model_id==spec['model_id'] and kwargs=={'revision':spec['revision'],'files_metadata':True}
            return hub_info(spec,digest)
    def download(request,root):
        events.append('bytes');path=root/request['revision'];path.mkdir()
        (path/request['weight_file']).write_bytes(content)
        (path/'config.json').write_text('{"model_type":"clip"}')
        return path
    monkeypatch.setattr('huggingface_hub.HfApi',Hub)
    monkeypatch.setattr(runner,'FROZEN_DOWNLOAD',download)
    directory=runner.download_model(spec,tmp_path)
    identity=runner.model_identity(spec,directory)
    assert events==['metadata','bytes'] and identity['weight_sha256']==digest
    assert identity['checksum_receipt_sha256']==identity['file_sha256'][runner.RECEIPT]
    (directory/spec['weight_file']).write_bytes(b'different weights')
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        runner.model_identity(spec,directory)


@pytest.mark.parametrize('fault',['revision','missing','checksum'])
def test_receipt_refuses_wrong_or_missing_authoritative_metadata(fault):
    spec=weight_spec();info=hub_info(spec,'b'*64)
    if fault=='revision':info.sha='c'*40
    if fault=='missing':info.siblings=[]
    if fault=='checksum':info.siblings[0].lfs.sha256='not a hash'
    with pytest.raises(ValueError):runner.metadata_receipt(spec,info)


def test_unknown_checksum_never_accepts_unqualified_local_weights(tmp_path):
    with pytest.raises(ValueError,match='receipt missing'):
        runner.model_identity(weight_spec(),tmp_path)
    receipt=runner.metadata_receipt(weight_spec(),hub_info(weight_spec(),'b'*64))
    receipt['revision']='c'*40
    (tmp_path/runner.RECEIPT).write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='different weights'):
        runner.model_identity(weight_spec(),tmp_path)


def test_frozen_callbacks_restored_when_inference_fails(tmp_path,monkeypatch):
    spec=weight_spec();args=SimpleNamespace(models=['m'],download=False,check=False,limit=2)
    paths={'models':{},'download_root':tmp_path}
    monkeypatch.setattr(runner,'protocol',lambda args:({'models':{'m':spec}},paths,{},None,None))
    before=(runner.frozen.Encoder,runner.frozen.model_identity,runner.frozen.download_model,runner.frozen.CODE)
    def fail(args):
        assert runner.frozen.Encoder is LiteratureEncoder
        assert runner.EXTRA_CODE[-1] in runner.frozen.CODE
        raise RuntimeError('inference failure')
    monkeypatch.setattr(runner.frozen,'run',fail)
    with pytest.raises(RuntimeError,match='inference failure'):runner.run(args)
    assert (runner.frozen.Encoder,runner.frozen.model_identity,
            runner.frozen.download_model,runner.frozen.CODE)==before


def test_summary_uses_resolved_output_directory(tmp_path,monkeypatch):
    profile=tmp_path/'paths.json';profile.write_text(json.dumps({'output_root':'wrong'}))
    args=SimpleNamespace(models=['m'],download=False,check=False,limit=None,paths=profile)
    paths={'models':{},'download_root':tmp_path/'downloads','output_root':tmp_path/'actual'}
    monkeypatch.setattr(runner,'protocol',lambda args:({'models':{'m':weight_spec()}},paths,{},None,None))
    monkeypatch.setattr(runner.frozen,'run',lambda args:[])
    calls=[]
    def summarize(profile,models,output):
        assert json.loads(profile.read_text())['output_root']==str(paths['output_root'])
        calls.append(output)
    monkeypatch.setattr('scripts.summarize_clip_router_prompt_ablation.summary',summarize)
    monkeypatch.setattr(runner,'compare_original',lambda *args:None)
    monkeypatch.setattr(runner,'threshold_reports',lambda *args:None)
    runner.run(args)
    assert calls==[paths['output_root']/'summary']


def test_default_new_models_are_pinned_and_scientific_namespace_is_approximate():
    cfg=json.loads(runner.CONFIG.read_text())
    assert runner.CONFIG.parent==runner.REPO/'configs/approximate'
    assert cfg['reproduction_mode']=='approximate' and cfg['table_5_claim'] is False
    assert cfg['execution']['batch_size']==2
    assert cfg['comparison_budget']=={'train':62,'test':149}
    for name in ('datacomp_l14','dfn_h14','siglip2_l16_384'):
        spec=cfg['models'][name]
        assert len(spec['revision'])==40
        assert spec['evidence_url'].startswith('https://arxiv.org/')
    assert cfg['models']['siglip2_l16_384']['text_preprocessing']=='lowercase_before_tokenizer'


def test_fast_gemma_from_json_runs_siglip_without_sentencepiece(tmp_path):
    """Native classes with a toy vocabulary: no pretrained checkpoint claim."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoTokenizer, SiglipConfig, SiglipModel
    backend=Tokenizer(models.WordLevel({'<pad>':0,'<eos>':1,'<bos>':2,'<unk>':3,
        'text':4,'written':5,'word':6},unk_token='<unk>'))
    backend.pre_tokenizer=pre_tokenizers.Whitespace();backend.save(str(tmp_path/'tokenizer.json'))
    (tmp_path/'tokenizer_config.json').write_text(json.dumps({'tokenizer_class':'GemmaTokenizer',
        'add_bos_token':False,'add_eos_token':True,'padding_side':'right',
        'pad_token':'<pad>','bos_token':'<bos>','eos_token':'<eos>','unk_token':'<unk>',
        'model_input_names':['input_ids']}))
    tokenizer=AutoTokenizer.from_pretrained(tmp_path,local_files_only=True,use_fast=True)
    assert tokenizer.is_fast and tokenizer.padding_side=='right'
    tokens=tokenizer(['text','text written word'],padding='max_length',max_length=8,
                     return_tensors='pt',truncation=False)['input_ids']
    assert tokens.tolist()==[[4,1,0,0,0,0,0,0],[4,5,6,1,0,0,0,0]]
    config=SiglipConfig(text_config={'vocab_size':7,'hidden_size':16,'intermediate_size':32,
        'num_attention_heads':4,'num_hidden_layers':1,'max_position_embeddings':8},
        vision_config={'hidden_size':16,'intermediate_size':32,'num_attention_heads':4,
            'num_hidden_layers':1,'image_size':8,'patch_size':2})
    encoder=object.__new__(LiteratureEncoder);encoder.model=SiglipModel(config).eval()
    encoder.tokenizer=LowercaseTokenizer(tokenizer);encoder.device='cpu'
    encoder.batch_size=2;encoder.limit=8;encoder.backend='siglip'
    embeddings=encoder.embeddings(['TEXT','TEXT written word'])
    assert embeddings.shape==(2,16) and np.isfinite(embeddings).all()
    np.testing.assert_allclose(np.linalg.norm(embeddings,axis=1),[1,1],atol=1e-6)


def test_threshold_curves_preserve_word_denominators_and_missing_fallback(tmp_path):
    from scripts.crop_clip_runtime import sha, write_json
    root=tmp_path/'m';root.mkdir()
    ids={'train':['a','b'],'test':['c','d']}
    cfg={'gt_words':{'train':2,'test':2},'report_thresholds':[0,.5,1]}
    plan=root/'resolved_configuration.json';write_json(plan,{'config':cfg});digest=sha(plan)
    selection={'winner':'native_only','families':{'native_only':{'threshold':.5}},
        'reference':{'native':[.1,.9],'word':{'baseline':[.1,.2],'margin':[0,.02]},
                     'scene':{'all_en':[.5]}}}
    write_json(root/'train_selection.json',selection)
    result={'result_class':'approximate reproduction','plan_sha256':digest,
            'train_selection_sha256':sha(root/'train_selection.json'),'split_ids':ids}
    write_json(root/'results.json',result)
    branches={}
    for split,sids in ids.items():
        branches[split]=[]
        for i,sid in enumerate(sids):
            f={'sample_id':sid,'plan_sha256':digest,'eligible':True,
               'native':[.1 if i==0 else .9],'word_feature_names':['baseline','margin'],
               'word_features':[{'baseline':.1,'margin':.01}],'scene_features':{'all_en':.5}}
            f['payload_sha256']=runner.payload_digest(f)
            write_json(root/split/(sid+'.json'),f)
            counts={'tp':1,'fp':0,'fn':0,'tn':0,'detected_words':1,'gt_words':1,
                    'all_outputs':1,'ignored_outputs':0,'invalid_envelopes':0}
            predicted={**counts,'tp':1-i,'fp':i,'fn':i}
            fallback=None if split=='train' and i==0 else counts
            branches[split].append({'sample_id':sid,'parseq':predicted,'deepsolo':fallback})
    outputs=[{'model':'m','results':str(root/'results.json')}]
    report=runner.threshold_reports(tmp_path,outputs,branches)
    test=report['models']['m']['splits']['test']['grid']
    assert [p['parseq_images'] for p in test]==[2,1,0]
    assert [p['words_retained_parseq'] for p in test]==[2,1,0]
    assert [p['words_from_deepsolo'] for p in test]==[0,1,2]
    assert test[-1]['combined_model']['accuracy_tn0']==1
    assert test[0]['combined_model']['accuracy_tn0']==pytest.approx(1/3)
    train=report['models']['m']['splits']['train']['grid']
    assert train[0]['combined_model'] is not None
    assert train[1]['combined_model'] is None and train[1]['missing_deepsolo_sample_ids']==['a']
    assert report['threshold_selected_on_test'] is False
    altered=json.loads((root/'test/c.json').read_text());altered['native']=[.5]
    write_json(root/'test/c.json',altered)
    with pytest.raises(ValueError,match='cache changed'):
        runner.threshold_reports(tmp_path,outputs,branches)
