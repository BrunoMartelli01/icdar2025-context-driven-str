"""Six approximate context-router experiments with immutable frozen base branches."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'

import numpy as np
from PIL import Image
import torch
from sklearn.model_selection import GroupKFold

from recognition.creative_context_signals import (
    ContextRenderer,context_views,expanded_job,feature_schemas,glyph_features,
    permutation_features,structure_features,swap_features,
)
from recognition.context_memory_router import fit_memory,predict_memory
from recognition.parseq import verify_parseq_runtime_source,PARSEQ_CHECKPOINT_SHA256,PARSEQ_CHECKPOINT_NAME
from recognition.trainable_image_router import branch_f1,oracle
from recognition.f1_quality_study import aggregate,evaluate
from scripts import run_trainable_image_router as old
from scripts.run_yolo_f1_quality_study import checked,read,seal,sha
from scripts.crop_clip_runtime import array_sha

REPO=Path(__file__).resolve().parents[1]
CODE=tuple(dict.fromkeys(('scripts/run_creative_context_router.py','recognition/creative_context_signals.py',
    'recognition/creative_context_runtime.py','recognition/context_memory_router.py','recognition/parseq.py',
    'recognition/grounded_context_runtime.py','recognition/residual_text_vqa.py',
    'scripts/contextual_lm_verifier_runtime.py','scripts/crop_clip_runtime.py',*old.CODE)))


def prepare(args):
    if args.mode!='approximate' or any(p.resolve().parent!=REPO/'configs/approximate' for p in (args.config,args.paths)):
        raise ValueError('Explicit isolated approximate configuration required')
    cfg,paths=read(args.config),read(args.paths)
    if cfg['reproduction_mode']!='approximate' or paths['reproduction_mode']!='approximate' or cfg['table_5_claim']:
        raise ValueError('Approximate non-Table5 metadata required')
    if cfg['feature_sets']!=['core','swap','permutation','glyph','structure','render','all'] or cfg['model_kinds']!=['ridge','trees']:
        raise ValueError('Six-method ablation contract differs')
    if cfg['memory_variants']!=['visual','joint'] or not cfg['memory_k'] or any(type(k)is not int or k<1 for k in cfg['memory_k']):
        raise ValueError('Explicit positive memory sizes and known variants required')
    if cfg['primary_family']!='trees_all' or cfg['cv_protocol']!='GroupKFold by original RGB SHA256, no shuffle':
        raise ValueError('Predeclared primary/CV protocol differs')
    if cfg['execution']['precision']!='float32' or not torch.cuda.is_available(): raise RuntimeError('Qualified FP32 CUDA required')
    source=Path(paths['source_root']).resolve();p=checked(source/'resolved_configuration.json',cfg['source_pins']['plan'])
    result=checked(source/'results.json',cfg['source_pins']['results']);audit=checked(source/'independent_audit.json',cfg['source_pins']['audit'])
    if result['status']!='SUCCESS' or audit['status']!='PASS' or audit['results_sha256']!=cfg['source_pins']['results']:
        raise ValueError('Complete independently audited source required')
    for name,digest in p['code_sha256'].items():
        if sha(source/'code_snapshot'/name)!=digest: raise ValueError('Source snapshot changed')
    fit,val=p['fit_ids'],p['validation_ids'];samples=p['samples']
    if args.smoke:
        fit=fit[:cfg['smoke_sizes']['fit']];val=val[:cfg['smoke_sizes']['validation']]
        samples={'train':[s for s in samples['train'] if s['sample_id'] in fit+val],'test':samples['test'][:cfg['smoke_sizes']['test']]}
    if set(fit)&set(val) or set(fit+val)!={s['sample_id'] for s in samples['train']}: raise ValueError('Invalid fitting partition')
    coverage=Path(p['source_coverage_root']);inputs={};observed={}
    for name,pin in [('resolved_configuration.json','plan'),('results.json','results'),('independent_audit.json','audit')]:inputs[str(source/name)]=cfg['source_pins'][pin]
    for split,ss in samples.items():
        observed[split]={}
        for sample in ss:
            sid=sample['sample_id'];key=sid.replace(':','__')+'.json';records={}
            for stage in ('features','inventory','verification'):
                path=source/split/stage/key;value=read(path)
                if value['plan_sha256']!=cfg['source_pins']['plan'] or value['image_sha256']!=sample['image_sha256']:
                    raise ValueError('Observable source identity changed')
                if stage=='features' and (value['per_image_ground_truth_used'] or value['actual_deepsolo_outputs_used']):
                    raise ValueError('Source inputs must precede actual fallback outcomes')
                inputs[str(path)]=sha(path);records[stage]=value
            rows_path=coverage/'640'/split/'features'/key;rows=read(rows_path)
            if rows['ground_truth_used'] or rows['image_sha256']!=sample['image_sha256']: raise ValueError('Original word input mismatch')
            if sha(Path(sample['image_path']))!=sample['image_sha256']:raise ValueError('Original RGB changed')
            inputs[str(rows_path)]=sha(rows_path)
            observed[split][sid]={'sample':sample,'rows':rows['rows'],'source':records}
    bp=coverage/'results.json';branches=read(bp)['resolutions']['640']['branch_metrics'];inputs[str(bp)]=sha(bp)
    if inputs[str(bp)]!=p['input_sha256'][str(bp)]: raise ValueError('Frozen branch counts changed')
    runtime=verify_parseq_runtime_source(Path(paths['parseq_source']),source_type='github_commit_archive',source_archive=Path(paths['parseq_archive']))
    if Path(paths['parseq_checkpoint']).name!=PARSEQ_CHECKPOINT_NAME or sha(Path(paths['parseq_checkpoint']))!=PARSEQ_CHECKPOINT_SHA256:
        raise ValueError('PARSeq checkpoint changed')
    for name in ('parseq_checkpoint','parseq_archive'):inputs[paths[name]]=sha(Path(paths[name]))
    asset=cfg['verifier_identity'];snapshot=Path(paths['verifier_snapshot'])
    if snapshot.name!=asset['revision']:raise ValueError('BLIP2 revision mismatch')
    for file,digest in {**asset['configuration_file_sha256'],**asset['weight_sha256']}.items():
        if sha(snapshot/file)!=digest:raise ValueError('BLIP2 artifact changed')
        inputs[str(snapshot/file)]=digest
    if len(paths['fonts'])!=len(cfg['font_sha256']):raise ValueError('Font bank size differs')
    for i,font in enumerate(paths['fonts']):
        if sha(Path(font))!=cfg['font_sha256'][str(i)]:raise ValueError('Font bytes changed')
        inputs[font]=cfg['font_sha256'][str(i)]
    output=Path(args.output_root or paths['output_root']).resolve()
    if any(output.is_relative_to(x) or x.is_relative_to(output) for x in (REPO,source,Path(paths['parseq_source']),snapshot)):
        raise ValueError('Use external outputs separate from sources/checkpoints')
    # Determine the schema without image labels or new visual inference.
    renderer=ContextRenderer(paths['fonts'],cfg['render_settings'])
    render,_=renderer.infer(np.zeros((1,1,3),np.uint8),[],cfg['geometry'])
    glyph,_=glyph_features([],[],cfg['glyph_settings'],cfg['geometry']);structure,_=structure_features([],[],cfg['geometry'])
    schemas=feature_schemas(p['feature_schemas']['core'],{'swap':swap_features([]),'permutation':permutation_features([]),
        'glyph':glyph,'structure':structure,'render':render})
    families=[kind+'_'+f for kind in cfg['model_kinds'] for f in cfg['feature_sets']]
    families+=['memory_'+v for v in cfg['memory_variants']]+['control_'+v for v in cfg['controls']]
    donors={};lookup=observed['train']
    for split,items in observed.items():
        donors[split]={}
        for sid,v in items.items():
            vector=np.array(v['source']['inventory']['embedding'])
            available=[t for t in fit if lookup[t]['sample']['image_sha256']!=v['sample']['image_sha256']]
            donors[split][sid]=min(available,key=lambda t:(float(vector@np.array(lookup[t]['source']['inventory']['embedding'])),t))
    plan={'schema_version':1,'reproduction_mode':'approximate','claim_label':'approximate reproduction','table_5_claim':False,
        'result_class':'SMOKE_TEST' if args.smoke else 'APPROXIMATE_EXPLORATORY_SIX_CONTEXT_METHODS',
        'config':cfg,'paths':paths,'config_sha256':sha(args.config),'paths_sha256':sha(args.paths),'samples':samples,
        'fit_ids':fit,'validation_ids':val,'families':families,'feature_schemas':schemas,'donor_ids':donors,
        'source_coverage_root':str(coverage),'metric':p['metric'],'input_sha256':inputs,
        'code_sha256':{f:sha(REPO/f) for f in CODE},'parseq_source_identity':runtime,
        'gpu':torch.cuda.get_device_name(0),'cuda_build':torch.version.cuda,'software':{n:importlib.metadata.version(n) for n in ('torch','numpy','scipy','scikit-learn','Pillow','transformers')},
        'base_models_frozen':True,'new_visual_model_inference':True,'new_neural_weight_updates':False,
        'historically_explored_test':True,'deep_solo_trained_on_totaltext_train':True,'deterministic_algorithms':True,'tf32':False}
    reuse_path=getattr(args,'reuse_probes_from',None)
    plan['reused_neural_probes']=None
    if reuse_path is not None:
        reuse_path=reuse_path.resolve();prior=read(reuse_path/'resolved_configuration.json')
        if prior['config']!=cfg or prior['samples']!=samples or prior['donor_ids']!=donors:
            raise ValueError('Raw neural reuse requires identical configuration/population/donors')
        for name in ('recognition/creative_context_runtime.py','recognition/parseq.py','recognition/grounded_context_runtime.py',
                     'recognition/residual_text_vqa.py','scripts/contextual_lm_verifier_runtime.py','scripts/crop_clip_runtime.py'):
            if prior['code_sha256'][name]!=plan['code_sha256'][name]:raise ValueError('Neural reuse runtime changed: '+name)
        def neural_functions(path):
            tree=ast.parse(path.read_text())
            return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)
                    and n.name in ('context_views','expanded_job','glyph_patch')}
        file='recognition/creative_context_signals.py'
        if neural_functions(reuse_path/'code_snapshot'/file)!=neural_functions(REPO/file):raise ValueError('Raw neural pixel/geometry functions changed')
        files={}
        for split,ss in samples.items():
            for kind in ('permutation','swap'):
                receipt=read(reuse_path/split/(kind+'_receipt.json'));m=receipt['model']
                if receipt['images']!=len(ss) or m['optimizer_steps']!=0 or m['model_state_sha256_before']!=m['model_state_sha256_after']:
                    raise ValueError('Only complete qualified frozen neural stages may be reused')
                paths_to_pin=[reuse_path/split/(kind+'_receipt.json')]+[reuse_path/split/kind/(s['sample_id'].replace(':','__')+'.json') for s in ss]
                files.update({str(p.relative_to(reuse_path)):sha(p) for p in paths_to_pin})
        for file in ('frozen_choices.json','cv_results.json'):files[file]=sha(reuse_path/file)
        plan['reused_neural_probes']={'root':str(reuse_path),'plan_sha256':sha(reuse_path/'resolved_configuration.json'),'files_sha256':files}
    tag=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()[:12]
    return cfg,plan,output/('smoke' if args.smoke else 'full')/tag,observed,branches


def image(sample):
    if sha(Path(sample['image_path']))!=sample['image_sha256']:raise ValueError('RGB checksum drift')
    with Image.open(sample['image_path']) as im:rgb=np.asarray(im.convert('RGB'))
    if list(rgb.shape[:2][::-1])!=sample['image_wh']:raise ValueError('Image orientation differs')
    return rgb


def probe_stage(cfg,plan,root,split,observed,kind):
    from recognition.creative_context_runtime import FrozenPermutationProbe,FrozenContextSwap
    ph=sha(root/'resolved_configuration.json');receipt=root/split/(kind+'_receipt.json')
    if receipt.exists():
        if read(receipt)['plan_sha256']!=ph:raise ValueError('Stage receipt drift')
        return
    if split=='test' and not (root/'frozen_choices.json').exists():raise ValueError('Freeze TRAIN choices before TEST probes')
    if plan.get('reused_neural_probes'):
        reuse=plan['reused_neural_probes'];source=Path(reuse['root'])
        binding=sha(root/'frozen_choices.json') if split=='test' else None
        for sid,value in observed[split].items():
            relative=split+'/'+kind+'/'+sid.replace(':','__')+'.json';raw=checked(source/relative,reuse['files_sha256'][relative])
            if raw['plan_sha256']!=reuse['plan_sha256'] or raw['sample_id']!=sid or raw['image_sha256']!=value['sample']['image_sha256']:
                raise ValueError('Reused raw probe identity differs')
            seal(root/relative,{**raw,'plan_sha256':ph,'frozen_choices_sha256':binding,'reused_source_sha256':reuse['files_sha256'][relative]})
        relative=split+'/'+kind+'_receipt.json';oldreceipt=checked(source/relative,reuse['files_sha256'][relative])
        seal(receipt,{**oldreceipt,'plan_sha256':ph,'seconds':0.,'reused_source_seconds':oldreceipt['seconds'],
            'reused_source_receipt_sha256':reuse['files_sha256'][relative]})
        print(f'[{kind} {split}] reused {len(observed[split])} qualified immutable probes',flush=True)
        return
    if kind=='permutation':
        paths={k:Path(plan['paths']['parseq_'+k]) for k in ('source','checkpoint','archive')}
        model=FrozenPermutationProbe(paths,cfg['permutation_settings'],cfg['execution']['device'])
    else:
        model=FrozenContextSwap(plan['paths']['verifier_snapshot'],device=cfg['execution']['device'],plain_question=cfg['question'],context_template='{caption}')
    start=time.monotonic()
    try:
        for i,(sid,v) in enumerate(observed[split].items(),1):
            path=root/split/kind/(sid.replace(':','__')+'.json');binding=sha(root/'frozen_choices.json') if split=='test' else None
            if path.exists():
                raw=read(path)
                if raw['plan_sha256']!=ph or raw['frozen_choices_sha256']!=binding:raise ValueError('Cached probe contract differs')
            else:
                rgb=image(v['sample'])
                if kind=='permutation':raw=model.infer(rgb,v['rows'])
                else:
                    donor_id=plan['donor_ids'][split][sid];donor=observed['train'][donor_id]['sample'];donor_rgb=image(donor);probes=[]
                    for prior in v['source']['verification']['probes']:
                        keys=('candidate_id','target_box','context_box','line_candidate_ids','alternatives','line_transcript')
                        job=expanded_job({k:prior[k] for k in keys},v['sample']['image_wh'],cfg['swap_settings'])
                        views=context_views(rgb,job,donor_rgb,cfg['swap_settings']);probe=model.infer(views,job,cfg['question'])
                        probe.update(donor_id=donor_id,donor_image_sha256=donor['image_sha256']);probes.append(probe)
                    raw={'probes':probes,'ground_truth_used':False,'base_outputs_changed':False}
                seal(path,{'plan_sha256':ph,'sample_id':sid,'image_sha256':v['sample']['image_sha256'],
                    'frozen_choices_sha256':binding,**raw})
            if i%10==0 or i==len(observed[split]):print(f'[{kind} {split}] {i}/{len(observed[split])}',flush=True)
        seal(receipt,{'plan_sha256':ph,'images':len(observed[split]),'seconds':time.monotonic()-start,'model':model.finish()})
    finally:model.close()


def features(cfg,plan,root,split,observed):
    renderer=ContextRenderer(plan['paths']['fonts'],cfg['render_settings']);ph=sha(root/'resolved_configuration.json');out={}
    for sid,v in observed.items():
        key=sid.replace(':','__')+'.json';perm=read(root/split/'permutation'/key);swap=read(root/split/'swap'/key)
        binding=sha(root/'frozen_choices.json') if split=='test' else None
        for raw in (perm,swap):
            if raw['plan_sha256']!=ph or raw['image_sha256']!=v['sample']['image_sha256'] or raw['frozen_choices_sha256']!=binding:
                raise ValueError('Raw probe binding differs')
        rgb=image(v['sample']);render,render_record=renderer.infer(rgb,v['rows'],cfg['geometry'])
        glyph,glyph_record=glyph_features(perm['words'],v['rows'],cfg['glyph_settings'],cfg['geometry'])
        labels=v['source']['inventory']['tasks']['<DENSE_REGION_CAPTION>']['parsed']['labels']
        structure,structure_record=structure_features(v['rows'],labels,cfg['geometry'])
        groups={'swap':swap_features(swap['probes']),'permutation':permutation_features(perm['words']),
            'glyph':glyph,'structure':structure,'render':render}
        schemas=feature_schemas(plan['feature_schemas']['core'],groups)
        if schemas!=plan['feature_schemas']:raise ValueError('Ablation feature schema changed')
        values={k:v['source']['features']['values'][k] for k in schemas['core']}
        for group in groups.values():values.update(group)
        if not np.isfinite(list(values.values())).all():raise ValueError('Nonfinite new context inputs')
        value={'plan_sha256':ph,'sample_id':sid,'image_sha256':v['sample']['image_sha256'],'values':values,
            'visual_embedding':v['source']['inventory']['embedding'],'eligible':v['source']['features']['eligible'],
            'controls':v['source']['features']['controls'],'per_image_ground_truth_used':False,'actual_deepsolo_outputs_used':False,
            'frozen_choices_sha256':binding,'evidence':{'render':render_record,'glyph':glyph_record,'structure':structure_record},
            'probe_sha256':{k:sha(root/split/k/key) for k in ('permutation','swap')}}
        seal(root/split/'features'/key,value);out[sid]=value
    return out


def recipes(family,cfg):
    if family.startswith('control_'):return [{'kind':'control','field':family[len('control_'):]}]
    if family.startswith('memory_'):return [{'kind':'memory','k':k,'variant':family.split('_')[1]} for k in cfg['memory_k']]
    return old.recipes(family.split('_')[0],cfg)


def fit_one(cfg,plan,features,branches,ids,recipe,family,fraction):
    if not set(ids)<=set(plan['fit_ids']):raise ValueError('Training outside declared fitting IDs')
    if recipe['kind']=='control':return {'recipe':recipe,'family':family,'training_ids':[]}
    if recipe['kind']!='memory':return old.train_one(cfg,plan,features,branches,ids,recipe,family,fraction)
    labels={s:branches[s] for s in ids}
    model=fit_memory(features,labels,ids,recipe['variant'],recipe['k'],plan['feature_schemas']['core'])
    model.update(recipe=recipe,family=family,reference_f1=float(branch_f1(labels,ids,[])))
    return model


def predict_ids(model,features,ids):
    kind=model['recipe']['kind']
    if kind=='control':return {s:features[s]['controls'][model['recipe']['field']] if features[s]['eligible'] else None for s in ids}
    if kind!='memory':return old.predict_ids(model,features,ids)
    p=predict_memory(model,features,ids);scores=2*p[:,0]-model['reference_f1']*p[:,1]
    return {s:float(v) if features[s]['eligible'] else None for s,v in zip(ids,scores)}


def fit_and_freeze(cfg,plan,root,features,branches):
    fit,val=plan['fit_ids'],plan['validation_ids'];rgb=[features[s]['image_sha256'] for s in fit]
    folds=list(GroupKFold(n_splits=cfg['cv_folds']).split(fit,groups=rgb));cv={};models={};calls=0
    frozen={'plan_sha256':sha(root/'resolved_configuration.json'),'fit_ids':fit,'validation_ids':val,'budgets':{},'models_sha256':{},
        'test_labels_used':False,'test_predictions_present_at_freeze':0,'test_features_present_at_freeze':0}
    for fraction in cfg['bypass_fractions']:
        key=str(fraction);cv[key]={};models[key]={};trials=[];cuts={};val_scores={}
        for index,family in enumerate(plan['families']):
            candidates=[]
            for recipe in recipes(family,cfg):
                oof={};trace=[]
                for a,b in folds:
                    train=[fit[i] for i in a];held=[fit[i] for i in b]
                    model=fit_one(cfg,plan,features,branches,train,recipe,family,fraction);calls+=recipe['kind']!='control'
                    oof.update(predict_ids(model,features,held));trace.append({'training_ids':train,'held_out_ids':held,
                        'reference_f1':model.get('reference_f1')})
                accepted=old.top(oof,fit,fraction);p=evaluate(branches,fit,accepted,cfg['quality_f1_threshold'])
                candidates.append({'recipe':recipe,'oof_scores':oof,'oof_point':p,'fold_trace':trace})
            winner=max(range(len(candidates)),key=lambda i:(branch_f1(branches,fit,candidates[i]['oof_point']['accepted_ids']),-i))
            cv[key][family]={'selected_trial':winner,'trials':candidates}
            model=fit_one(cfg,plan,features,branches,fit,candidates[winner]['recipe'],family,fraction);calls+=model['recipe']['kind']!='control'
            path=root/'models'/(key+'__'+family+'.json');seal(path,model);frozen['models_sha256'][str(path.relative_to(root))]=sha(path)
            models[key][family]=model;ss=predict_ids(model,features,val);val_scores[family]=ss
            accepted=old.top(ss,val,fraction);cuts[family]=min((ss[s] for s in accepted),default=None)
            trials.append((branch_f1(branches,val,accepted),-index,family))
        frozen['budgets'][key]={'selected_family':max(trials)[2],'scalar_cuts':cuts,
            'validation_f1':{family:float(score) for score,_,family in trials}}
        print(f'[fit {fraction}] VAL selected {max(trials)[2]}',flush=True)
    seal(root/'cv_results.json',cv);frozen.update(cv_results_sha256=sha(root/'cv_results.json'),fit_calls=calls)
    if plan.get('reused_neural_probes'):
        reuse=plan['reused_neural_probes'];prior=checked(Path(reuse['root'])/'frozen_choices.json',reuse['files_sha256']['frozen_choices.json'])
        for key in ('models_sha256','budgets','cv_results_sha256'):
            if frozen[key]!=prior[key]:raise ValueError('Neural-cache replay changed fitted models/choices: '+key)
        frozen['neural_reuse_models_and_choices_identical']=True
    if not (root/'frozen_choices.json').exists() and list((root/'test').rglob('*.json')):
        raise ValueError('Cannot select after new TEST probes')
    seal(root/'frozen_choices.json',frozen);print('[frozen] TRAIN/VAL choices before new TEST evidence',flush=True)
    return frozen,models


def evaluate_split(cfg,plan,features,branches,ids,models,frozen):
    out={'images':len(ids),'light_only':aggregate(branches[s]['light'] for s in ids),
         'deepsolo_only':aggregate(branches[s]['deepsolo'] for s in ids),'budgets':{}};predictions={}
    eligible=[s for s in ids if features[s]['eligible']];references={}
    def point(accepted):
        n=len(accepted)
        if n not in references:references[n]=oracle({s:branches[s] for s in ids},ids,eligible,n)
        p=evaluate(branches,ids,accepted,cfg['quality_f1_threshold'])
        p.update(oracle_reference_f1=references[n]['f1'],oracle_f1_gap=references[n]['f1']-p['combined']['f1'])
        return p
    for fraction in cfg['bypass_fractions']:
        key=str(fraction);predictions[key]={};families={};scalar={}
        for family,model in models[key].items():
            ss=predict_ids(model,features,ids);predictions[key][family]=ss;accepted=old.top(ss,ids,fraction)
            families[family]=point(accepted);cut=frozen['budgets'][key]['scalar_cuts'][family]
            scalar[family]={'threshold':cut,**point([s for s in ids if cut is not None and ss[s] is not None and ss[s]>=cut])}
        controls={name:families['control_'+name] for name in cfg['controls']};chosen=frozen['budgets'][key]['selected_family']
        k=round(len(ids)*fraction);point([])
        if k not in references:references[k]=oracle({s:branches[s] for s in ids},ids,eligible,k)
        out['budgets'][key]={'families':families,'scalar_thresholds':scalar,'controls':controls,
            'selected_family':chosen,'selected':families[chosen],'primary_family':cfg['primary_family'],'primary':families[cfg['primary_family']],
            'oracle':references[k]}
    return out,predictions


def run(args):
    cfg,plan,root,observed,branches=prepare(args)
    print(json.dumps({'status':'PREFLIGHT_PASS','root':str(root),'families':len(plan['families']),
        'feature_counts':{k:len(v) for k,v in plan['feature_schemas'].items()}}),flush=True)
    if args.preflight:return
    seal(root/'resolved_configuration.json',plan)
    for file,digest in plan['code_sha256'].items():
        target=root/'code_snapshot'/file;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha(target)!=digest:raise ValueError('Immutable code snapshot drift')
        if not target.exists():target.write_bytes((REPO/file).read_bytes())
    torch.set_num_threads(cfg['execution']['cpu_threads']);seed=cfg['execution']['seed'];random.seed(seed);np.random.seed(seed)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        for split in ('train','test'):
            for kind in ('permutation','swap'):probe_stage(cfg,plan,root,split,observed,kind)
            values=features(cfg,plan,root,split,observed[split])
            if split=='train':train=values;frozen,models=fit_and_freeze(cfg,plan,root,train,branches['train'])
            else:test=values
        result={'status':'SUCCESS','reproduction_mode':'approximate','claim_label':'approximate reproduction','table_5_claim':False,
            'result_class':plan['result_class'],'plan_sha256':sha(root/'resolved_configuration.json'),
            'frozen_choices_sha256':sha(root/'frozen_choices.json'),'base_models_frozen':True,'original_word_outputs_changed':False,
            'new_visual_model_inference':True,'router_training':True,'splits':{}}
        for split,values,b,ids in [('fit',train,branches['train'],plan['fit_ids']),('validation',train,branches['train'],plan['validation_ids']),
            ('train',train,branches['train'],sorted(train)),('test',test,branches['test'],sorted(test))]:
            result['splits'][split],predictions=evaluate_split(cfg,plan,values,b,ids,models,frozen)
            seal(root/'predictions'/(split+'.json'),predictions)
        result['paired_intervals']=old.intervals(cfg,result,branches['test']);seal(root/'results.json',result)
        print(json.dumps({'status':'SUCCESS','root':str(root),'results_sha256':sha(root/'results.json')}),flush=True)
    except BaseException as error:
        value={'status':'FAILED','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
               'plan_sha256':sha(root/'resolved_configuration.json')}
        file='failure_'+hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:12]+'.json';seal(root/file,value)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--mode',required=True,choices=['approximate'])
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--paths',type=Path,required=True)
    parser.add_argument('--output-root',type=Path);parser.add_argument('--preflight',action='store_true');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--reuse-probes-from',type=Path,help='Explicit identical qualified neural source; require unchanged fitted models and choices')
    run(parser.parse_args())
