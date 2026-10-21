import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from recognition.clip_specialized_runtime import (EncoderBridge, RemoteTokenizer,
    NativeEncoder, caption_mean_pool, llama_caption_parts,
    clip_rgb_pixels, longclip_dependency_compatibility, normalized_vectors)


@pytest.mark.parametrize('text',['dog','  ÉCOLE  ','all the text present is: A B','\t東京\n'])
def test_llama_framing_matches_published_marker_then_strip_semantics(text):
    marker = '!@#$%^&*()'
    published = '<|start_header_id|>user<|end_header_id|>\n\n' + (marker+text).strip() + '<|eot_id|>'
    prefix,body = published.split(marker)
    framed,pool_body = llama_caption_parts(text)
    assert framed == prefix+body
    assert pool_body == body


def test_llama_delimiter_collision_fails_instead_of_dropping_characters():
    with pytest.raises(ValueError,match='delimiter'):
        llama_caption_parts('abc!@#$%^&*()def')


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_body_mean_pool_ignores_header_and_padding_and_retains_eot(dtype):
    hidden = torch.arange(2*7*3,dtype=torch.float32).reshape(2,7,3).to(dtype)
    pooled = caption_mean_pool(hidden,[3,1])
    expected = torch.stack([hidden[0,-3:].mean(0),hidden[1,-1:].mean(0)])
    assert torch.equal(pooled,expected)
    changed = hidden.clone();changed[0,:4] = -999
    assert torch.equal(caption_mean_pool(changed,[3,1]),expected)


@pytest.mark.parametrize('lengths',[[0,1],[8,1],[1]])
def test_invalid_pool_masks_fail(lengths):
    with pytest.raises(ValueError,match='pooling'):
        caption_mean_pool(torch.ones(2,7,3),lengths)


def test_normalization_keeps_float32_and_rejects_invalid_embeddings():
    result = normalized_vectors(torch.tensor([[3.,4.]],dtype=torch.bfloat16))
    assert result.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(result,axis=1),[1.],atol=1e-6)
    for invalid in [torch.zeros(1,2),torch.tensor([[float('nan'),1.]])]:
        with pytest.raises(ValueError,match='embedding'):
            normalized_vectors(invalid)


def test_native_token_ids_are_cached_without_changing_order_or_ids():
    calls=[]
    def request(payload):
        calls.append(payload)
        return {'input_ids':[49406,123,49407]}
    tokenizer=RemoteTokenizer(SimpleNamespace(request=request))
    for _ in range(2):
        assert tokenizer('word',add_special_tokens=True,truncation=False)['input_ids']==[49406,123,49407]
    assert len(calls)==1
    with pytest.raises(ValueError,match='untruncated'):
        tokenizer('word',add_special_tokens=True,truncation=True)


def test_native_failure_propagates_traceback_and_memory_receipt():
    bridge=object.__new__(EncoderBridge);bridge.failed=False
    payload={'schema_version':1,'success':False,'error':'CUDA out of memory',
             'memory':{'peak_reserved_bytes':123}}
    bridge.process=SimpleNamespace(stdout=io.StringIO(json.dumps(payload)+'\n'))
    with pytest.raises(RuntimeError,match='out of memory'):
        bridge.read_response()
    assert bridge.failed and bridge.memory['peak_reserved_bytes']==123


def test_native_response_population_order_is_checked():
    bridge=object.__new__(EncoderBridge);bridge.failed=False;bridge.counter=0
    bridge.process=SimpleNamespace(stdin=io.StringIO())
    bridge.read_response=lambda:{'request_id':99,'memory':{}}
    with pytest.raises(ValueError,match='ordering'):
        bridge.request({'op':'token_ids','text':'word'})
    assert bridge.failed


def test_cuda_initializes_before_legacy_allocator_statistics(monkeypatch):
    events=[]
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'init',lambda:events.append('init'))
    def reset(device):
        assert events==['init']
        events.append('reset')
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',reset)
    identity={'batch_size':2,'context_length':248,'backend':'unknown',
              'precision':'float32','file_sha256':{}}
    with pytest.raises(ValueError,match='Unsupported'):
        NativeEncoder(identity,'cuda:0')
    assert events==['init','reset']


def test_longclip_compatibility_changes_only_obsolete_dependency_reexport():
    original = ('from pkg_resources import packaging\n'
                'result = packaging.version.parse("1.13.1")\n')
    qualified = longclip_dependency_compatibility(original)
    namespace = {}
    exec(qualified, namespace)
    assert str(namespace['result']) == '1.13.1'
    assert qualified.splitlines()[1:] == original.splitlines()[1:]
    with pytest.raises(ValueError, match='import changed'):
        longclip_dependency_compatibility('import packaging\n')


def test_close_propagates_already_exited_nonzero_worker_and_cleans_resources():
    bridge=object.__new__(EncoderBridge);bridge.failed=False
    bridge.process=SimpleNamespace(poll=lambda:7,returncode=7)
    bridge.log=io.StringIO()
    cleaned=[]
    bridge.temporary=SimpleNamespace(cleanup=lambda:cleaned.append(True))
    with pytest.raises(RuntimeError,match='nonzero'):
        bridge.close()
    assert bridge.log.closed and cleaned==[True]


def test_tiny_rgb_crops_use_known_channel_order_in_mixed_batches():
    from PIL import Image
    from transformers import CLIPImageProcessor
    processor=CLIPImageProcessor(size={'shortest_edge':8},crop_size={'height':8,'width':8})
    color=np.array([10,120,230],dtype=np.uint8)
    images=[Image.fromarray(np.broadcast_to(color,(n,n,3)).copy(),'RGB') for n in (3,2,1,5)]
    batched=clip_rgb_pixels(processor,images)
    assert batched.shape==(4,3,8,8)
    expected=(color/255-np.array(processor.image_mean))/np.array(processor.image_std)
    np.testing.assert_allclose(batched[:,:,0,0].numpy(),np.tile(expected,(4,1)),atol=1e-6)
    for index,image in enumerate(images):
        assert torch.equal(batched[index],clip_rgb_pixels(processor,[image])[0])
