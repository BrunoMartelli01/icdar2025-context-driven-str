"""Join qualified shared-pipeline experiments without selecting TEST methods."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

from recognition.budget_target_oracle import analyze_budget


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


RECIPE_FIELDS=['seed','folds','correction_confidence_caps','unmatched_policies','reference_grid',
    'guards','word_risk','feature_groups','head_recipes','ensemble_sizes','selection_tolerance','map_features']
CORE_FUNCTIONS=['normalize','counts','micro','choose_light','route','grouped_folds','map_features',
    'geometry_features','word_features','risk_features','image_features','fit_count_head','predict_counts','fit_word_risk']


def qualify(root):
    q=read(root/'deployment_qualification.json')
    if q['status']!='PASS' or sha(root/q['audit_path'])!=q['audit_sha256']:raise ValueError('Unqualified report')
    for name,digest in q['artifact_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Qualified artifact changed: '+name)
    return read(root/q['audit_path']),Path(q['audit_path']).parent


def report(main,ic13,criteria,output):
    roots=[main,ic13];plans=[read(r/'resolved_configuration.json') for r in roots]
    if any(p['smoke'] or p['mode']!='approximate' for p in plans):raise ValueError('Complete approximate runs required')
    recipes=[{k:p['config'][k] for k in RECIPE_FIELDS} for p in plans]
    if recipes[0]!=recipes[1]:raise ValueError('Training recipes differ')
    ast_hashes=[]
    for r in roots:
        nodes=ast.parse((r/'code_snapshot/recognition/common_budget_pipeline.py').read_text()).body
        functions={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in nodes if isinstance(n,ast.FunctionDef) and n.name in CORE_FUNCTIONS}
        if set(functions)!=set(CORE_FUNCTIONS):raise ValueError('Incomplete common core')
        ast_hashes.append(functions)
    if ast_hashes[0]!=ast_hashes[1]:raise ValueError('Scientific core functions changed')
    acceptance=read(criteria);rows=[];oracles={};cases={};inventory={};audits={};pins={str(criteria):sha(criteria)}
    for root,plan in zip(roots,plans):
        audit,folder=qualify(root);audits[str(root)]=audit;data=read(root/'results.json');summary=read(root/folder/'summary.json');metadata=read(root/'cases.json')
        pins[str(root/'results.json')]=sha(root/'results.json');pins[str(root/'deployment_qualification.json')]=sha(root/'deployment_qualification.json')
        for name,value in data['results'].items():
            dataset,version=name.split('__',1);point=value['primary'];counts=point['combined'];f1=counts['f1'];baseline=value['baseline_f1'];meta=metadata[name]
            if dataset=='totaltext':target=acceptance['totaltext']['minimum_f1'];met=f1>target
            elif dataset=='icdar2015':target=acceptance[dataset]['minimum_f1_by_version'][version];met=f1>=target-1e-12
            else:target=plan['config']['budgets'][dataset]['minimum_f1'];met=f1>=target-1e-12
            rows.append(dict(dataset=dataset,version=version,f1_percent=100*f1,baseline_f1_percent=100*baseline,
                difference_points=100*(f1-baseline),target_f1_percent=100*target,target_met=bool(met),
                deepsolo_images=point['deepsolo_images'],total_images=point['total_images'],deepsolo_percentage=point['deepsolo_percentage']))
            if dataset=='icdar13':
                source=read(root/'test'/(name+'.json'))
                ideal=analyze_budget(source['branches'],list(source['branches']),point['deepsolo_images'],str(target))
                if ideal['exact_budget']['optimality_residual'] not in (0,0.,'0'):raise ValueError('Uncertified image oracle')
                oracles[name]=ideal
            selected=read(root/'models'/name/'selection_frozen.json');models=selected['winner']['models'];modeldir=root/'models'/name
            image_bundles=list(modeldir.glob('head_*.joblib'));word_bundles=list(modeldir.glob('word_*.joblib'))
            inventory[name]=dict(image_head_bundles_trained=len(image_bundles),word_models_trained=len(word_bundles),selected_image_head_bundles=len(models),
                selected_inference_files={m:dict(bytes=(modeldir/m).stat().st_size,sha256=sha(modeldir/m)) for m in models+['word_full.joblib']},
                selected_groups=selected['winner']['groups'],reference=selected['winner']['reference'],guard=selected['winner']['guard'])
            cases[name]=dict(root=str(root),audit_directory=str(folder),counts=counts,paired_bootstrap=summary[name]['bootstrap'],
                deployment_score_max_error=summary[name]['deployment_score_max_absolute_error'])
    if len(rows)!=7:raise ValueError('Missing dataset/version outcomes')
    output.mkdir(exist_ok=False)
    write(output/'comparison.json',rows)
    with (output/'comparison.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(output/'icdar13_gt_oracles.json',dict(scope='Privileged TEST-GT image choice, never a deployed selector; same newly selected expert outputs',results=oracles))
    common_met=all(v['target_met'] for v in rows if v['dataset']!='icdar13')
    recipe_sha=hashlib.sha256(json.dumps(recipes[0],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write(output/'common_method_identity.json',dict(recipe=recipes[0],recipe_sha256=recipe_sha,scientific_core_ast_sha256=ast_hashes[0],
        validation='Separate selection/CAL, unchanged. Historical first source predates its later explicit configuration switch; recorded roles prove separate execution.',
        scope='Identical correction/features/learning/selection/quota core around qualified domain-specific expert and vocabulary inputs'))
    write(output/'model_inventory.json',inventory)
    write(output/'summary.json',dict(status='COMPLETE',mode='approximate',claim_label='approximate reproduction',common_tt_ic15_user_criteria_met=common_met,
        icdar13_any_version_reached90=any(v['target_met'] for v in rows if v['dataset']=='icdar13'),results=rows,cases=cases,
        actual_new_neural_inference=0,actual_new_deepsolo_inference=0,historical_test_exposure=True,
        total_new_image_head_bundles=sum(v['image_head_bundles_trained'] for v in inventory.values()),total_new_word_models=sum(v['word_models_trained'] for v in inventory.values()),
        recipe_sha256=recipe_sha,source_inputs_sha256=pins,producer_sha256=sha(__file__)))
    lines=['# Shared budget pipeline results','','Approximate reproduction. All complete TEST images/care words retained. Exact logical decisions on real cached expert outputs; no online latency measurement.','',
        '| Dataset | Version | F1 % | Previous F1 % | Difference, points | User target met | DeepSolo images/total (%) |','| --- | --- | ---: | ---: | ---: | --- | --- |']
    for v in rows:lines.append(f"| {v['dataset']} | {v['version']} | {v['f1_percent']:.4f} | {v['baseline_f1_percent']:.4f} | {v['difference_points']:+.4f} | {v['target_met']} | {v['deepsolo_images']}/{v['total_images']} ({v['deepsolo_percentage']:.2f}%) |")
    lines+=['','User revised TT acceptance to strictly above78.2%; IC15 must retain its previous F1. Original small TT losses remain shown. The mixed IC13 version runs YOLO and a U-Net complement.','',
        '| ICDAR2013 version | Actual F1 % | GT-perfect F1 % | DeepSolo images/total (%) |','| --- | ---: | ---: | --- |']
    for name,v in oracles.items():
        point=v['exact_budget'];actual=next(r for r in rows if 'icdar13__'+r['version']==name)
        lines.append(f"| {name} | {actual['f1_percent']:.4f} | {100*point['f1']:.4f} | {point['deepsolo_images']}/{point['total_images']} ({point['deepsolo_percent']:.2f}%) |")
    lines+=['','Identical recipe and scientific core-function AST hashes were verified across the TT/IC15 and IC13 source snapshots. Source adapters, trained expert weights/resolutions, allowed vocabularies and optimized DeepSolo artifacts remain explicit inputs. Historical TEST exploration/upstream fitting exposure and benchmark vocabulary side information limit generalization claims. Bootstrap intervals condition on fitted choices.','',
        'Deployment: `CommonBudgetPipeline(root, case).score(rows, image_wh, probabilities)` followed by `route_batch(observations, rgb_hashes, fallback_images=K)`. Predictions consume no GT/care masks/fallback outputs. The caller invokes the qualified dataset-specific DeepSolo backend for exactly the returned fallback IDs.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='COMPLETE',common_targets_met=common_met,report=str(output),rows=rows)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--main-root',type=Path,required=True);p.add_argument('--icdar13-root',type=Path,required=True);p.add_argument('--criteria',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();report(a.main_root,a.icdar13_root,a.criteria,a.output)
