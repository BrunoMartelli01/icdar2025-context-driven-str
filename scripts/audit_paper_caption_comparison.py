"""Independent count, routing, cache and semantic audit for the caption study."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def stable(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def independent_counts(rows,truth,ignored):
    def normalize(s):return ''.join(c for c in s.casefold() if c.isalnum())
    assert len(rows)==len(ignored)
    texts=[normalize(r['raw_text']) for r,skip in zip(rows,ignored) if not skip]
    expected=Counter(normalize(r['text']) for r in truth if normalize(r['text']))
    correct=sum((Counter(s for s in texts if s)&expected).values())
    return dict(tp=correct,fp=len(texts)-correct,fn=len(truth)-correct,
                detected_words=len(texts),gt_words=len(truth),all_outputs=len(rows),ignored_outputs=sum(ignored))


def aggregate(rows):
    rows=list(rows);v={k:sum(r[k] for r in rows) for k in rows[0]}
    den=2*v['tp']+v['fp']+v['fn']
    return dict(v,images=len(rows),f1=2*v['tp']/den if den else 0.)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--reencode',action='store_true');args=p.parse_args()
    root=args.root;plan=read(root/'resolved_configuration.json');report=read(root/'results.json')
    scores=read(root/'per_image_scores.json');assert plan['mode']==report['mode']=='approximate'
    for path,h in plan['source_sha256'].items():assert sha(path)==h,path
    for path,h in plan['config']['pinned_inputs'].items():assert sha(path)==h,path
    asset_files_checked=0
    for asset in plan['assets'].values():
        if 'snapshot' not in asset:continue
        snapshot=Path(asset['snapshot'])
        weights=asset.get('weight_sha256')
        weights=weights if isinstance(weights,dict) else {'model.safetensors':weights}
        for relative,h in dict(weights,**asset.get('configuration_file_sha256',{})).items():
            assert sha(snapshot/relative)==h,str(snapshot/relative)
            asset_files_checked+=1
    caches={};peaks={};sample_count=0
    for path in sorted((root/'cache').glob('*/*.json')):
        d=read(path);c=d['contract'];sid=c['sample']['sample_id'];stage=c['stage'];payload=d['payload']
        assert d['payload_sha256']==stable(payload)
        assert c['sample']==plan['samples'][sid]
        assert c['source_sha256']==plan['source_sha256'] and c['runtime']==plan['runtime']
        assert c['plan_config_sha256']==plan['config_sha256'] and c['assets']==plan['assets']
        assert sha(c['sample']['image_path'])==c['sample']['image_sha256']
        expected_batch = plan['config'].get('caption_batch_size', 1) if stage == 'captions' else 1
        assert 1 <= payload['batch_size'] <= expected_batch
        if stage == 'captions' and expected_batch > 1:
            assert len(payload['shared_batch_ids']) == payload['batch_size']
        assert 0<payload['peak_allocated_bytes']<=payload['peak_reserved_bytes']
        if stage=='captions':
            result=payload['result'];assert result['image_only_captioning'] and not result['ground_truth_in_caption'] and not result['parseq_in_prompt']
            assert result['profile']==plan['config']['caption_profile']
            assert result['answers'][0]['result']['generation_config']==plan['config']['caption_profile']['generation']
            assert result['raw_caption']==result['answers'][0]['result']['raw_caption']
        else:assert payload['result']['input_role']=='T1_full_image'
        caches[sid,stage]=payload
        peaks[stage]={k:max(peaks.get(stage,{}).get(k,0),payload[k]) for k in ('peak_allocated_bytes','peak_reserved_bytes')}
        sample_count+=1
    assert sample_count==2*len(plan['samples'])
    semantic_pairs=[];counts_checked=0;policies_checked=0;decision_checks=0
    from rapidfuzz.fuzz import ratio
    for name,case in plan['cases'].items():
        source=Path(case['source_run']);labels=read(source/'data'/(name+'__test_labels.json'))
        outputs=read(source/'models'/name/'observable_outputs.json')['corrected_rows']
        light={};t1={};fallback={}
        for sid,rows in case['rows'].items():
            assert rows==outputs[sid]
            f=scores[name][sid];raw1=caches[sid,'t1']['result']['raw_text'];caption=caches[sid,'captions']['result']['raw_caption']
            order=sorted(enumerate(rows),key=lambda pair:(pair[1]['bbox'][1],pair[1]['bbox'][0],f'row:{pair[0]:06d}'))
            # Raw output bytes and empty readings remain in the space join.
            transcript=' '.join(r['raw_text'] for _,r in order)
            assert f['texts']==dict(T1=raw1,T2=caption,T3=transcript)
            if f['context_overflow']:
                assert max(f['token_lengths'].values())>384 and all(x is None for x in f['scores'].values())
            else:
                d=f['decision'];winner='T1' if d['S1']>d['S3'] else 'T3'
                l1=ratio(raw1,caption)/100;l3=ratio(transcript,caption)/100
                c1=.6*d['S1']+.4*l1;c3=.6*d['S3']+.4*l3
                assert d['semantic_winner']==winner
                for key,value in dict(L1=l1,L3=l3,C1=c1,C3=c3,C=c1 if winner=='T1' else c3).items():
                    assert abs(d[key]-value)<1e-12,(name,sid,key)
                eligible3=bool(rows) and all(r['raw_text'].strip() for r in rows) and bool(caption.strip())
                eligible=bool(caption.strip()) and (bool(raw1.strip()) if winner=='T1' else eligible3)
                assert f['scores']['paper_selected']==(d['C'] if eligible else None)
                assert f['scores']['joined_t3']==(c3 if eligible3 else None)
                semantic_pairs.append((f['texts'],d));decision_checks+=1
            if sid not in case['partitions']['test']:continue
            label=labels[sid]
            light[sid]=independent_counts(rows,label['ground_truth'],[label['ignored'][r['source_row']] for r in rows])
            t1[sid]=independent_counts([dict(raw_text=raw1)],label['ground_truth'],[False])
            fallback[sid]=label['teacher'];counts_checked+=2
        result=report['results'][name]
        assert result['light_branches']==light and result['t1_branches']==t1 and result['fallback_branches']==fallback
        original=read(source/'test'/(name+'.json'))
        for method,point in result['policies'].items():
            if method=='all_light':accepted=set(light);gate='joined_t3'
            elif method=='all_deepsolo':accepted=set();gate='joined_t3'
            elif method=='trained_lightweight':
                accepted=set(original['accepted_ids'])&set(light);gate='joined_t3'
            else:
                gate=method.split('__')[0];values={s:scores[name][s]['scores'][gate] for s in light}
                rule=point['rule']
                if 'threshold' in rule:accepted={s for s,v in values.items() if v is not None and v>=rule['threshold']}
                else:
                    n=len(light)-rule['fallback_images']
                    accepted=set(sorted((s for s,v in values.items() if v is not None),key=lambda s:(-values[s],case['tie_keys'][s],s))[:n])
                assert set(point['accepted_ids'])==accepted
            chosen={s:(t1[s] if gate=='paper_selected' and scores[name][s]['decision']['semantic_winner']=='T1' else light[s]) if s in accepted else fallback[s] for s in light}
            assert aggregate(chosen.values())==point['combined'],(name,method)
            assert point['deepsolo_images']==len(light)-len(accepted)
            assert point['total_images']==len(light)
            assert point['deepsolo_percentage']==100*(len(light)-len(accepted))/len(light)
            if 'per_image' in point:assert point['per_image']==chosen
            policies_checked+=1
    max_error=None
    if args.reencode:
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(plan['config']['cpu_threads'])
        model=SentenceTransformer(plan['assets']['mpnet']['snapshot'],device='cpu',local_files_only=True,
                                  model_kwargs={'torch_dtype':torch.float32});model.max_seq_length=384
        texts=sorted({text for f,d in semantic_pairs for text in f.values()})
        vectors=model.encode(texts,batch_size=32,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
        vectors={s:np.asarray(v,np.float64) for s,v in zip(texts,vectors)};max_error=0.
        def cosine(a,b):return float(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))
        for f,d in semantic_pairs:
            for text,field in [('T1','S1'),('T3','S3')]:
                error=abs(cosine(vectors[f[text]],vectors[f['T2']])-d[field]);max_error=max(max_error,error)
                assert error<=2e-6,(text,error)
    result=dict(status='PASS',result_sha256=sha(root/'results.json'),resolved_configuration_sha256=sha(root/'resolved_configuration.json'),
                auditor_sha256=sha(__file__),cache_shards=sample_count,independent_word_count_records=counts_checked,
                independently_replayed_policies=policies_checked,independent_score_decisions=decision_checks,
                semantic_reencoding=args.reencode,max_semantic_error=max_error,semantic_tolerance=2e-6,
                model_weight_and_processor_files_checked=asset_files_checked,
                gpu_peaks=peaks,new_deepsolo_inference=0)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True)
    print(json.dumps(result))


if __name__=='__main__':main()
