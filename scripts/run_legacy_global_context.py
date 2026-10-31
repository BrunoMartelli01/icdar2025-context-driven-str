"""Add image-wide context to the previously qualified word-LM feature baseline."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from recognition.contextual_boundary_selection import class_utility,select
from scripts.run_contextual_boundary_selection import evaluate,bootstrap
from scripts.run_text_reconstruction_tests import read,write,sha

REPO=Path(__file__).resolve().parents[1]


def run(config,output,mode):
    config=Path(config).resolve();output=Path(output)
    if mode!='approximate' or REPO/'configs/approximate' not in config.parents:raise ValueError('Explicit APPROXIMATE configuration required')
    c=read(config)
    if c['reproduction_mode']!=mode:raise ValueError('Mode mismatch')
    for p,expected in c['input_sha256'].items():
        if sha(p)!=expected:raise ValueError('Pinned legacy input changed: '+p)
    source=Path(c['source_run']);old=Path(c['legacy_run'])
    if not (source/'completion.json').exists():raise ValueError('Complete primary evidence is required')
    data=read(old/'candidates.json');pool=read(source/'pool.json');labels=read(old/'fit_labels.json')
    cfg=read(old/'resolved_config.json');settings=cfg['selector'];features={}
    for phase,samples in data.items():
        original=read(old/f'features_{phase}.json');lm=read(source/f'language_{phase}.json');by={s['sample_id']:s for s in pool[phase]};features[phase]={}
        for s in samples:
            sid=s['sample_id'];p=by[sid]
            if p['blocks']!=s['blocks']:raise ValueError('Legacy and primary observations differ')
            indices={(tuple(o['parents']),o['text'].casefold()):i for i,o in enumerate(p['pool'])};rows=[]
            for i,options in enumerate(s['candidates']):
                variants=[]
                for j,o in enumerate(options[1:]):
                    key=((i,),o['text'].casefold())
                    if key not in indices:raise ValueError('Legacy candidate absent from the primary scored pool')
                    evidence=lm[sid]['operations'][indices[key]];word=original[sid][i][j]['word_lm']
                    variants.append(dict(legacy_word=word,legacy_global=word+evidence['local']+evidence['ordered']+evidence['inventory']))
                rows.append(variants)
            features[phase][sid]=rows
    output.mkdir(parents=True,exist_ok=False);write(output/'resolved_config.json',c);models={};scores={};selection={}
    y=np.array([label['label'] for s in data['fit'] for group in labels[s['sample_id']] for label in group])
    for arm in ('legacy_word','legacy_global'):
        x=np.array([variant[arm] for s in data['fit'] for group in features['fit'][s['sample_id']] for variant in group])
        model=HistGradientBoostingClassifier(max_iter=settings['iterations'],max_leaf_nodes=settings['max_leaf_nodes'],
            min_samples_leaf=settings['min_samples_leaf'],learning_rate=settings['learning_rate'],l2_regularization=settings['l2_regularization'],
            early_stopping=False,random_state=settings['seeds'][0]).fit(x,y)
        models[arm]=model;scores[arm]={}
        for phase in ('calibration','evaluation','test'):
            scores[arm][phase]={}
            for s in data[phase]:
                rows=[]
                for group in features[phase][s['sample_id']]:
                    rows.append(class_utility(model.predict_proba(np.array([r[arm] for r in group])),model.classes_).tolist() if group else [])
                scores[arm][phase][s['sample_id']]=rows
        curve=[]
        for threshold in settings['thresholds']:
            pred={s['sample_id']:select(s,scores[arm]['calibration'][s['sample_id']],threshold)[0] for s in data['calibration']}
            curve.append(dict(threshold=threshold,**evaluate(data['calibration'],pred)['aggregate']))
        choices={str(cap):max([r for r in curve if r['damaged']<=cap],key=lambda r:(r['correct'],r['f1'],r['threshold'])) for cap in settings['damage_limits']}
        selection[arm]=dict(curve=curve,choices=choices,dimensions=x.shape[1]);print('LEGACY_CALIBRATED',arm,json.dumps(choices['2']),flush=True)
    write(output/'selection_frozen.json',dict(models=selection,evaluation_test_used_for_selection=False));write(output/'scores.json',scores)
    with (output/'models.pkl').open('wb') as stream:pickle.dump(models,stream)
    results={};predictions={};comparisons={};old_predictions=read(old/'predictions.json');checks=0
    for phase in ('calibration','evaluation','test'):
        arms={}
        for arm in models:
            for cap,choice in selection[arm]['choices'].items():
                arms[arm+'_cap'+cap]={s['sample_id']:select(s,scores[arm][phase][s['sample_id']],choice['threshold'])[0] for s in data[phase]}
        expected=old_predictions[phase][f"word_lm_seed{settings['seeds'][0]}_cap2"]
        if arms['legacy_word_cap2']!=expected:raise ValueError('Legacy baseline replay differs; do not certify context comparison')
        checks+=len(data[phase]);predictions[phase]=arms;results[phase]={name:evaluate(data[phase],p) for name,p in arms.items()}
        if phase!='calibration':comparisons[phase]=bootstrap(results[phase]['legacy_word_cap2'],results[phase]['legacy_global_cap2'],c['bootstrap_draws'],c['seed'])
    write(output/'predictions.json',predictions)
    write(output/'results.json',dict(status='SUCCESS',reproduction_mode='approximate',claim_label='approximate reproduction',results=results,comparisons=comparisons,
        selection_sha256=sha(output/'selection_frozen.json'),baseline_replay_images=checks,
        routing={p:dict(images_sent_to_deepsolo=0,total_evaluated_images=len(data[p]),percentage=0.,execution='offline saved PARSeq transcription') for p in results},
        limitation='Predeclared supplemental test on historically explored cohorts; no independent-validation or strict-paper claim.'))
    write(output/'independent_audit.json',dict(status='PASS',baseline_replay_images=checks,results_sha256=sha(output/'results.json')))
    print(json.dumps(dict(status='SUCCESS',baseline_replay_images=checks)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',required=True);parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();run(args.config,args.output,args.mode)


if __name__=='__main__':main()
