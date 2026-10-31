"""Verify native Qwen caption tokens, images, artifacts, prompts and provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--role',required=True,choices=['development','qualification','paper_example'])
    p.add_argument('--profiles',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();plan=read(args.root/('plan_'+args.role+'.json'));c=plan['config'];protocol=read(c['sample_protocol'])
    for f,h in plan['source_sha256'].items():assert sha(f)==h,f
    assert sha(c['sample_protocol'])==plan['sample_protocol_sha256']
    ids=(protocol['roles']['paper_example'] if args.role=='paper_example' else
         [s for v in protocol['roles'].values() if isinstance(v,dict) for s in v[args.role]])
    asset=c['assets']['qwen'];snapshot=Path(c['hf_cache'])/('models--'+asset['model_id'].replace('/','--'))/'snapshots'/asset['revision']
    for f,h in asset['weight_sha256'].items():assert sha(snapshot/f)==h,f
    from transformers import AutoProcessor
    import torch
    processor=AutoProcessor.from_pretrained(snapshot,local_files_only=True)
    checked=0;peaks=dict(allocated=0,reserved=0);profiles={}
    for name in args.profiles:
        profile=c['profiles'][name];assert profile['caption_model']=='qwen'
        paths=list((args.root/name/args.role).glob('*.json'));assert len(paths)==len(ids)
        words=[];observed=[]
        for path in paths:
            d=read(path);sid=d['sample']['sample_id'];observed.append(sid)
            assert d['sample']==protocol['samples'][sid] and sha(d['sample']['image_path'])==d['sample']['image_sha256']
            assert d['plan_sha256']==digest(plan) and d['profile']==profile
            result=d['result'];assert d['payload_sha256']==digest(result)
            assert result['image_only_captioning'] and not result['parseq_in_prompt'] and not result['ground_truth_in_caption']
            assert result['caption_family']=='Qwen2-VL-7B caption ablation'
            assert len(result['answers'])==1 and result['answers'][0]['prompt']==profile['prompt']
            native=result['answers'][0]['result'];assert native['generation_config']==profile['generation']
            assert native['prompt']==profile['prompt'] and native['min_pixels']==profile['min_pixels'] and native['max_pixels']==profile['max_pixels']
            assert native['dtype']=='bfloat16' and native['quantization']=='none'
            assert native['placement']==profile.get('placement','single_device') and native['attention_implementation']=='sdpa'
            if native['placement']=='cpu_offload':
                devices=list(native['hf_device_map'].values());assert 'cpu' in devices
                assert any(v=='0' or v.startswith('cuda') for v in devices) and 'disk' not in devices
            assert len(native['generated_token_ids'])<=profile['generation']['max_new_tokens']
            decoded=processor.batch_decode(torch.tensor([native['generated_token_ids']]),skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
            assert decoded==native['raw_caption']==result['raw_caption']
            assert len(decoded.split())==result['quality']['words']
            assert len(decoded)==result['quality']['characters']
            grid=native['image_grid_thw'];assert len(grid)==1 and all(x>0 for x in grid[0])
            visual_tokens=grid[0][0]*grid[0][1]*grid[0][2]/4
            assert profile['min_pixels']/28**2<=visual_tokens<=profile['max_pixels']/28**2
            assert 0<native['peak_allocated_bytes']<=native['peak_reserved_bytes']
            peaks['allocated']=max(peaks['allocated'],native['peak_allocated_bytes']);peaks['reserved']=max(peaks['reserved'],native['peak_reserved_bytes'])
            words.append(len(decoded.split()));checked+=1
        assert sorted(observed)==sorted(ids)
        profiles[name]=dict(images=len(words),mean_words=sum(words)/len(words),minimum_words=min(words),maximum_words=max(words))
    result=dict(status='PASS',role=args.role,profiles=profiles,native_suffix_token_replays=checked,
                image_and_prompt_records_checked=checked,weight_shards_checked=len(asset['weight_sha256']),gpu_peaks=peaks,
                source_sha256=sha(__file__),plan_sha256=sha(args.root/('plan_'+args.role+'.json')),
                factuality_scope='Token/provenance verification only; separate direct visual review required')
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True)
    print(json.dumps(result))


if __name__=='__main__':main()
