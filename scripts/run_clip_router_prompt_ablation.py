"""APPROXIMATE CLIP/SigLIP2 prompt/router comparisons on frozen epoch61 OCR.

No segmentation, OCR or DeepSolo model is rerun. All image words enter the gate.
Prompt choice uses TRAIN45; percentile references and 50% cut use TRAIN80.
TEST budget comparisons are descriptive, never used for selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import time
import traceback

import numpy as np
from PIL import Image
import torch

from recognition.clip_image_router import partition, accepted
from recognition.clip_router_prompt_ablation import (mutations, prompt_chunks, transcript,
    fit_reference, image_scores, coverage_threshold, budget_accept)
from recognition.clip_router_threshold_sweep import point, validate_population
from recognition.clip_word_verifier import unit
from recognition.group_word_selection_runtime import square_canvas
from scripts.crop_clip_runtime import sha, write_json, _cuda_peak
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.run_clip_high_recall_admission import payload_digest, seal

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO/'configs/approximate/clip_router_prompt_ablation_v2.json'
PATHS = REPO/'configs/approximate/clip_router_prompt_ablation_local_paths_v2.json'
CODE = ('scripts/run_clip_router_prompt_ablation.py', 'recognition/clip_router_prompt_ablation.py',
        'recognition/clip_router_threshold_sweep.py', 'recognition/clip_router_evaluation.py',
        'recognition/clip_image_router.py', 'recognition/clip_crop_admission.py',
        'recognition/clip_word_verifier.py', 'recognition/group_word_selection_runtime.py',
        'scripts/crop_clip_runtime.py', 'scripts/run_clip_high_recall_admission.py',
        'scripts/evaluate_totaltext_highres_contour_padding.py')


def checked(path, digest):
    if sha(path) != digest:
        raise ValueError('Frozen artifact changed: '+str(path))
    return json.loads(path.read_text())


def load_protocol(args):
    for path in (args.config, args.paths):
        if path.resolve().parent != REPO/'configs/approximate':
            raise ValueError('Configuration must be isolated in configs/approximate')
    cfg, paths = json.loads(args.config.read_text()), json.loads(args.paths.read_text())
    if (args.mode != 'approximate' or cfg['reproduction_mode'] != 'approximate'
            or cfg['claim_label'] != 'approximate reproduction' or cfg['table_5_claim'] is not False
            or cfg['execution']['precision'] != 'float32' or paths['path_base'] != 'repository_root'
            or not 0 < cfg['coverage'] <= 1):
        raise ValueError('Invalid approximate prompt protocol')
    for group in cfg['word_templates'].values():
        if not group or any(t.count('{text}') != 1 for t in group):
            raise ValueError('Invalid word templates')
    paths = {k: v for k, v in paths.items() if k not in ('schema_version','path_base')}
    for k in ('source_run','branch_metrics','output_root','download_root'):
        paths[k] = Path(paths[k]).resolve()
    if args.output_root:
        paths['output_root'] = args.output_root.resolve()
    if paths['output_root'].is_relative_to(REPO) or paths['output_root'].is_relative_to(paths['source_run']):
        raise ValueError('An isolated external output directory is required')
    source = checked(paths['source_run']/'resolved_configuration.json', cfg['source_plan_sha256'])
    for name, digest in source['code_sha256'].items():
        if sha(REPO/name) != digest:
            raise ValueError('Historical source code changed: '+name)
    branches = checked(paths['branch_metrics'],cfg['branch_metrics_sha256'])['splits']
    for split, rows in branches.items():
        validate_population(rows, source['split_ids'][split], cfg['gt_words'][split])
    observed, input_hashes = {}, {}
    for split in ('train','test'):
        observed[split] = []
        for sid in source['split_ids'][split]:
            path = paths['source_run']/split/'routes'/(sid.replace(':','__')+'.json')
            r = json.loads(path.read_text());s=r['observed']
            if (r['payload_sha256'] != payload_digest(r) or r['contract'] != {
                    'plan_sha256':cfg['source_plan_sha256'], 'sample_id':sid,'image_sha256':s['image_sha256']}
                    or sha(Path(s['image_path'])) != s['image_sha256']):
                raise ValueError('Frozen OCR/image route changed')
            if len(s['rows']) != next(x for x in branches[split] if x['sample_id']==sid)['parseq']['all_outputs']:
                raise ValueError('OCR population differs from audited branch metrics')
            observed[split].append(s);input_hashes[str(path)]=sha(path)
    return cfg, paths, observed, branches, input_hashes


def model_identity(spec, directory):
    if not re.fullmatch('[0-9a-f]{40}',spec['revision']) or directory.name != spec['revision']:
        raise ValueError('Full model revision/directory required')
    digest = sha(directory/spec['weight_file'])
    if digest != spec['weight_sha256']:
        raise ValueError('Model checkpoint SHA256 mismatch')
    hashes={p.name:sha(p) for p in directory.iterdir() if p.is_file()
            and (p.suffix in ('.json','.txt','.model') or p.name==spec['weight_file'])}
    config=json.loads((directory/'config.json').read_text())
    if config['model_type']!=spec['backend']:
        raise ValueError('Checkpoint/backend mismatch')
    return {**spec,'local_path':str(directory),'file_sha256':hashes}


def download_model(spec, root):
    from huggingface_hub import snapshot_download
    directory=root/spec['revision']
    snapshot_download(repo_id=spec['model_id'],revision=spec['revision'],local_dir=directory,
        allow_patterns=[spec['weight_file'],'*.json','*.txt','*.model','README.md'])
    return directory


class Encoder:
    def __init__(self, identity, device, batch_size):
        from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
        directory=identity['local_path'];self.device=device;self.batch_size=batch_size
        self.model,info=AutoModel.from_pretrained(directory,local_files_only=True,
            use_safetensors=identity['weight_file'].endswith('safetensors'),output_loading_info=True)
        if any(info.values()):
            raise ValueError('Checkpoint key mismatch: '+str(info))
        self.model=self.model.eval().float().to(device)
        self.tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,use_fast=True)
        self.processor=AutoImageProcessor.from_pretrained(directory,local_files_only=True)
        self.backend=identity['backend'];self.limit=self.model.config.text_config.max_position_embeddings
        self.image_size=self.model.config.vision_config.image_size
        if self.image_size != identity['image_size']:
            raise ValueError('Model input resolution differs')
        self.cache={}

    def embeddings(self, values, images=False):
        output=[]
        for i in range(0,len(values),self.batch_size):
            batch=values[i:i+self.batch_size]
            if images:
                inputs=self.processor(images=batch,return_tensors='pt',input_data_format='channels_last')
                kwargs={'pixel_values':inputs['pixel_values'].to(self.device)}
                fn=self.model.get_image_features
            else:
                if any(len(self.tokenizer(v,add_special_tokens=True,truncation=False)['input_ids'])>self.limit for v in batch):
                    raise ValueError('Refusing silent text truncation')
                token_options={'padding':True}
                if self.backend=='siglip':token_options={'padding':'max_length','max_length':self.limit}
                inputs=self.tokenizer(batch,return_tensors='pt',truncation=False,**token_options)
                kwargs={k:v.to(self.device) for k,v in inputs.items()}
                # SigLIP was trained without attention masking and with fixed padding.
                if self.backend=='siglip':
                    kwargs={'input_ids':kwargs['input_ids']}
                fn=self.model.get_text_features
            with torch.inference_mode():
                vector=fn(**kwargs)
                vector=torch.nn.functional.normalize(vector.float(),dim=-1)
            v=vector.cpu().numpy()
            if not np.isfinite(v).all():
                raise ValueError('Nonfinite embedding')
            output.extend(v)
        return np.asarray(output,dtype=np.float32)

    def ensure(self, prompts):
        missing=sorted(set(prompts)-self.cache.keys())
        if missing:
            self.cache.update(zip(missing,self.embeddings(missing)))

    def mean(self, prompts):
        return unit(np.stack([self.cache[p] for p in prompts]).mean(axis=0))


def features(encoder, cfg, observed):
    rows=observed['rows'];eligible=bool(rows) and all(r['raw_text'].strip() for r in rows)
    words={};prompts=list(cfg['negative_prompts'])
    for row in rows:
        text=row['raw_text']
        if not text.strip():continue
        for name,templates in cfg['word_templates'].items():
            words[text,name]=[p for t in templates for p in prompt_chunks(t,text,encoder.tokenizer,encoder.limit)]
            prompts.extend(words[text,name])
        for alternative in mutations(text):
            words[alternative,'mutation']=[p for t in cfg['word_templates']['baseline']
                for p in prompt_chunks(t,alternative,encoder.tokenizer,encoder.limit)]
            prompts.extend(words[alternative,'mutation'])
    whole=transcript(rows)
    scenes={name:prompt_chunks(t,whole,encoder.tokenizer,encoder.limit)
            for name,t in cfg['scene_templates'].items()}
    prompts.extend(p for chunks in scenes.values() for p in chunks)
    encoder.ensure(prompts);negative=encoder.mean(cfg['negative_prompts'])
    with Image.open(observed['image_path']) as image:rgb=np.asarray(image.convert('RGB'))
    imgs=[square_canvas(rgb)];active=[]
    for i,row in enumerate(rows):
        x1,y1,x2,y2=row['bbox'];crop=np.ascontiguousarray(rgb[y1:y2,x1:x2])
        if crop_array_hash(crop)!=row['crop_rgb_sha256']:
            raise ValueError('Frozen original crop RGB differs')
        if row['raw_text'].strip():imgs.append(square_canvas(crop));active.append(i)
    vectors=encoder.embeddings(imgs,images=True)
    word_features=[]
    for i,v in zip(active,vectors[1:]):
        text=rows[i]['raw_text'];value={name:float(v@(encoder.mean(words[text,name])-negative))
                                     for name in cfg['word_templates']}
        raw=float(v@encoder.mean(words[text,'baseline']))
        controls={a:float(v@encoder.mean(words[a,'mutation'])) for a in mutations(text)}
        value['margin']=raw-max(controls.values())
        word_features.append(value)
    # Every scene chunk contributes; use the weakest chunk cosine.
    scene_values={name:min(float(vectors[0]@encoder.cache[p]) for p in chunks)
                  for name,chunks in scenes.items()}
    return {'sample_id':observed['sample_id'],'eligible':eligible,
        'word_feature_names':[*cfg['word_templates'],'margin'],
        'native':[rows[i]['confidence'] for i in active], 'word_features':word_features,
        'scene_features':scene_values,'transcript':whole,'scene_prompt_chunks':scenes,
        'spelling_controls':{r['raw_text']:mutations(r['raw_text']) for r in rows},
        'ground_truth_in_scoring':False,'routing_includes_ignored_words':True}


def evaluate(rows, scores, threshold, ids, gt):
    replay=[{**r,'score':scores[r['sample_id']]} for r in rows]
    result=point(replay,threshold,ids,gt)
    selected=[r for r in rows if accepted(scores[r['sample_id']],threshold)]
    result['safe_images']=sum(r['parseq']['detected_words']>0 and r['parseq']['fp']==0 for r in selected)
    result['complete_images']=sum(r['parseq']['detected_words']>0 and r['parseq']['fp']==0 and r['parseq']['fn']==0 for r in selected)
    return result


def run(args):
    cfg,paths,observed,branches,input_hashes=load_protocol(args)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable here; execute on GPU host or select CPU')
    seed=cfg['execution']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(cfg['execution']['cpu_threads']);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    outputs=[]
    for name in args.models:
        spec=cfg['models'][name]
        directory=Path(paths['models'].get(name,paths['download_root']/spec['revision'])).resolve()
        if not (directory/spec['weight_file']).exists():
            if not args.download:
                raise FileNotFoundError(f'{name} unavailable. Execute --download on the network/GPU host.')
            directory=download_model(spec,paths['download_root'])
        identity=model_identity(spec,directory)
        plan={'schema_version':2,'reproduction_mode':'approximate','claim_label':'approximate reproduction',
            'result_class':'smoke-test result' if args.limit else 'approximate reproduction','table_5_claim':False,
            'config':cfg,'model':identity,'source_sha256':input_hashes,'branch_metrics_sha256':cfg['branch_metrics_sha256'],
            'config_sha256':sha(args.config),'code_sha256':{n:sha(REPO/n) for n in CODE},
            'device':args.device,'limit':args.limit,'software':{'torch':torch.__version__,
                'transformers':__import__('transformers').__version__,'numpy':np.__version__,
                'pillow':Image.__version__,'python':__import__('sys').version},
            'hardware':{'gpu':torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None,
                'cuda_build':torch.version.cuda},'deterministic_algorithms':True,'tf32':False}
        root=paths['output_root']/name/hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()[:12]
        seal(root/'resolved_configuration.json',plan);digest=sha(root/'resolved_configuration.json')
        if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(args.device)
        encoder=None;fs={};metrics={};ids={};by_family={}
        try:
            for split in ('train','test'):
                samples=observed[split][:args.limit] if args.limit else observed[split]
                ids[split]=[s['sample_id'] for s in samples]
                metrics[split]=[r for r in branches[split] if r['sample_id'] in ids[split]]
                for i,s in enumerate(samples,1):
                    path=root/split/(s['sample_id'].replace(':','__')+'.json')
                    if path.exists():
                        feature=json.loads(path.read_text())
                        if feature['plan_sha256']!=digest or feature['payload_sha256']!=payload_digest(feature):
                            raise ValueError('Feature cache changed')
                    else:
                        start=time.monotonic()
                        if encoder is None:encoder=Encoder(identity,args.device,cfg['execution']['batch_size'])
                        feature=features(encoder,cfg,s);feature['plan_sha256']=digest
                        feature['runtime']={'seconds':time.monotonic()-start,'device':args.device,
                            'precision':'float32','batch_size':cfg['execution']['batch_size'],
                            'image_shape':[3,spec['image_size'],spec['image_size']],
                            'models':[spec['model_id']], 'cuda_peak':_cuda_peak(args.device),'oom':False}
                        feature['payload_sha256']=payload_digest(feature);seal(path,feature)
                    fs[s['sample_id']]=feature
                    if i%25==0 or i==len(samples):print(f'[{name} {split}] {i}/{len(samples)}',flush=True)
                if split=='train':
                    partitioned=partition(ids['train'],cfg['calibration_images'],seed) if not args.limit else {
                        'calibration':ids['train'],'validation':ids['train']}
                    reference=fit_reference(fs,partitioned['calibration'])
                    train_scores={sid:image_scores(fs[sid],reference) for sid in ids['train']}
                    selections={}
                    for family in next(iter(train_scores.values())):
                        scores={sid:v[family] for sid,v in train_scores.items()}
                        threshold=coverage_threshold(scores,partitioned['calibration'],cfg['coverage'])
                        rows=[r for r in metrics['train'] if r['sample_id'] in partitioned['validation']]
                        val=evaluate(rows,scores,threshold,partitioned['validation'],sum(r['parseq']['gt_words'] for r in rows))
                        selections[family]={'threshold':threshold,'validation':val}
                    # Fixed objective: retained-word precision on TRAIN45; higher retained-word count breaks ties.
                    winner=max(selections,key=lambda f:(selections[f]['validation']['precision_retained_parseq'] or -1,
                        selections[f]['validation']['words_retained_parseq'],f))
                    selection={'fit_ids':partitioned['calibration'],'selection_ids':partitioned['validation'],
                        'reference':reference,'families':selections,'winner':winner,
                        'selection_uses_test_labels':False,'model_selection_is_exploratory_train':True}
                    seal(root/'train_selection.json',selection)
                for sid in ids[split]:
                    for family,value in image_scores(fs[sid],reference).items():
                        by_family.setdefault(family,{})[sid]=value
            comparisons={}
            for family,scores in by_family.items():
                comparisons[family]={}
                for split in ('train','test'):
                    gt=sum(r['parseq']['gt_words'] for r in metrics[split])
                    transferred=evaluate(metrics[split],scores,selections[family]['threshold'],ids[split],gt)
                    count=min(cfg['comparison_budget'][split],len(ids[split]))
                    keep=budget_accept(scores,ids[split],count)
                    exact={sid:float(sid in keep) for sid in ids[split]}
                    matched=evaluate(metrics[split],exact,1.,ids[split],gt)
                    comparisons[family][split]={'train_threshold':transferred,'same_budget_descriptive':matched}
            original={split:evaluate(metrics[split],{r['sample_id']:r['score'] for r in metrics[split]},
                cfg['original_threshold'],ids[split],sum(r['parseq']['gt_words'] for r in metrics[split])) for split in ('train','test')}
            result={'status':'SUCCESS','reproduction_mode':'approximate','claim_label':'approximate reproduction',
                'result_class':plan['result_class'],'plan_sha256':digest,'table_5_claim':False,
                'model':name,'winner_train':winner,'train_selection_sha256':sha(root/'train_selection.json'),
                'comparisons':comparisons,'original_router':original,'split_ids':ids,
                'selection_uses_test_labels':False,'same_budget_test_is_descriptive':True,
                'elapsed_seconds':sum(f['runtime']['seconds'] for f in fs.values()),'cuda_peak':_cuda_peak(args.device)}
            seal(root/'results.json',result);write_json(paths['output_root']/name/'latest.json',{'results_path':str(root/'results.json'),'results_sha256':sha(root/'results.json')})
            outputs.append({'model':name,'results':str(root/'results.json'),'winner_train':winner})
            print(json.dumps(outputs[-1]),flush=True)
        except Exception as error:
            write_json(root/'failure.json',{'status':'FAILED','error':str(error),'traceback':traceback.format_exc(),
                'out_of_memory':isinstance(error,(MemoryError,torch.cuda.OutOfMemoryError)),'cuda_peak':_cuda_peak(args.device)})
            raise
        finally:
            del encoder
            if args.device.startswith('cuda'):torch.cuda.empty_cache()
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=['approximate'])
    p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--paths',type=Path,default=PATHS)
    p.add_argument('--models',nargs='+',default=['vit_b16'])
    p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    p.add_argument('--download',action='store_true');p.add_argument('--output-root',type=Path)
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.limit is not None and args.limit<1:p.error('--limit must be positive')
    run(args)


if __name__=='__main__':main()
