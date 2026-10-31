"""Retrain native ICDAR gates against Generic / TRAIN-only Weak teachers."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from recognition.clip_router_evaluation import deepsolo_predictions
from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.deepsolo_lexicon_deployment import DeepSoloVocabularyPostprocessor
from recognition.deepsolo_word_f1 import apply_policy,recognition_confidence
from recognition.icdar_budget_router import lexical_counts
from recognition.f1_quality_study import aggregate
from scripts.icdar_budget_data import read,sha,key,annotation,predictions
from scripts.run_yolo_f1_quality_study import seal


def independent(pp,gt):
    def norm(s):return ''.join(c for c in s.casefold() if c.isalnum())
    care=[p for p in pp if not p['ignored']]
    a=Counter(norm(p['text']) for p in care if norm(p['text']));b=Counter(norm(g['text']) for g in gt if norm(g['text']))
    tp=sum((a&b).values())
    return dict(tp=tp,fp=len(care)-tp,fn=len(gt)-tp,detected_words=len(care),gt_words=len(gt),all_outputs=len(pp),ignored_outputs=len(pp)-len(care))


def checked_config(path):
    c=read(path)
    if c['mode']!='approximate' or c['claim_label']!='approximate reproduction':raise ValueError('Approximate mode required')
    for p,h in c['pinned_inputs'].items():
        if sha(p)!=h:raise ValueError('Changed frozen input: '+p)
    return c


def teacher(c,ds):
    root=Path(c['study_root']);old=Path(c['original_router']);plan=read(old/'resolved_configuration.json');spec=c['teacher'][ds]
    folder=root/'teacher'/ds;folder.mkdir(parents=True,exist_ok=True);records=plan['records'][ds]
    pins={};entries={};vocabulary=None
    if ds=='icdar13':
        profile=read(spec['profile']);model=DeepSoloVocabularyPostprocessor(profile)
        checkpoint=profile['checkpoint_sha256'];neural=profile['neural'];policy=profile['unmatched_policy']
        for p in profile['lexicon_paths'].values():pins[p]=sha(p)
        source_protocol=read(Path(spec['raw_train'])/'protocol.json')
    else:
        canonical={}
        for r in records['train']:
            gt,_=annotation(r);pins[r['annotation_path']]=r['annotation_sha256']
            for g in gt:canonical.setdefault(g['text'].strip().upper(),g['text'].strip())
        words=sorted(canonical)
        if any(not w for w in words):raise ValueError('Empty raw TRAIN vocabulary word')
        vocabulary=folder/'train_weak_vocabulary.txt';pairs=folder/'train_weak_pairs.json'
        with vocabulary.open('x') as f:f.write('\n'.join(words)+'\n')
        with pairs.open('x') as f:json.dump(canonical,f,indent=2);f.write('\n')
        pins[str(vocabulary)],pins[str(pairs)]=sha(vocabulary),sha(pairs)
        # TRAIN annotations can contain literal spaces. The original released
        # pair-list reader uses the first space as a delimiter; JSON preserves
        # the complete intended key without splitting or dropping any GT word.
        lexicon=DeepSoloLexicon(words,read(pairs),distance_exclusive=1.5)
        source_protocol=read(Path(spec['raw_existing_validation'])/'protocol.json')
        checkpoint=source_protocol['source_identity']['checkpoint_sha256'];neural=spec['neural'];policy=spec['unmatched_policy']
        new_protocol=read(Path(spec['raw_new_train'])/'protocol.json')
        if new_protocol['source_identity']['checkpoint_sha256']!=checkpoint:
            raise ValueError('New TRAIN exports use a different checkpoint')
        pins[str(Path(spec['raw_new_train'])/'protocol.json')]=sha(Path(spec['raw_new_train'])/'protocol.json')
    if source_protocol['source_identity']['checkpoint_sha256']!=checkpoint:raise ValueError('Wrong teacher checkpoint')
    print(ds,'teacher checkpoint',checkpoint,'TRAIN vocabulary size',len(words) if ds=='icdar2015' else len(model.lexicon.words),flush=True)
    samples={}
    for i,r in enumerate(records['train'],1):
        sid=r['sample_id'];gt,validity=annotation(r);pins[r['annotation_path']]=r['annotation_sha256']
        path=Path(spec['raw_train'])/(key(sid)+'.json') if ds=='icdar13' else Path(spec['raw_existing_validation'])/(key(sid)+'.json')
        if not path.exists() and ds=='icdar2015':path=Path(spec['raw_new_train'])/(key(sid)+'.json')
        if not path.exists():raise ValueError('New teacher inference still missing: '+sid)
        raw=read(path);body={k:v for k,v in raw.items() if k!='payload_sha256'}
        if (raw['payload_sha256']!=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
                or raw['contract']['image_sha256']!=r['image_sha256'] or raw['contract']['sample_id']!=sid or not raw['response']['success']):raise ValueError('Teacher raw identity failed')
        pins[str(path)]=sha(path)
        if ds=='icdar13':
            corrected=model.apply(dict(raw['response'],artifact=dict(checkpoint_sha256=checkpoint)));detections=corrected['detections'];decisions=corrected['lexicon_decisions']
        else:
            by={d['source_index']:d for d in raw['response']['detections']}
            if len(by)!=len(raw['response']['detections']):raise ValueError('Duplicate teacher source index')
            rows=[dict(instance_id=i,text=d['text'],score=d['score'],recognition_confidence=recognition_confidence(d['raw']['recognition_class_probabilities'])) for i,d in by.items()]
            rows=apply_policy(rows,dict(neural,resolution='native',decoder='greedy',same_text_nms_iou=None));rows,decisions=lexicon.correct(rows,unmatched_policy=policy)
            detections=[dict(by[p['instance_id']],text=p['text']) for p in rows]
        pp=deepsolo_predictions(dict(detections=detections),r['image_wh'],r['orientation'],validity,neural['score_threshold'])
        for p in pp:
            if p['invalid_envelope'] is not None:p['text']=''
        counts=lexical_counts(pp,gt)
        if counts!=independent(pp,gt):raise ValueError('Teacher word recount failed')
        dest=folder/'per_image'/(key(sid)+'.json');seal(dest,dict(sample_id=sid,predictions=pp,ground_truth=gt,counts=counts,decisions=decisions,raw_path=str(path),raw_sha256=pins[str(path)]))
        samples[sid]=dict(image_sha256=r['image_sha256'],counts=counts,evidence_path=str(dest),evidence_sha256=sha(dest))
        if i%100==0 or i==len(records['train']):print(ds,'teacher counts',i,'/',len(records['train']),flush=True)
    for r in records['test']:
        path=Path(c['replay_root'])/'fallback_evaluation'/ds/(key(r['sample_id'])+'.json');data=read(path)
        if data['checkpoint_sha256']!=checkpoint or independent(data['predictions'],data['ground_truth'])!=data['counts']:raise ValueError('Qualified TEST teacher recount failed')
        samples[r['sample_id']]=dict(image_sha256=r['image_sha256'],counts=data['counts'],evidence_path=str(path),evidence_sha256=sha(path))
    seal(folder/'counts.json',dict(status='PASS',mode='approximate',claim_label='approximate reproduction',dataset=ds,checkpoint_sha256=checkpoint,
        neural=neural,unmatched_policy=policy,train_vocabulary=str(vocabulary),train_vocabulary_scope='Only official TRAIN care words; zero TEST transcriptions or vocabulary used' if ds=='icdar2015' else 'Previously qualified released Generic dictionary',
        input_sha256=pins,samples=samples))


def prepare(c,ds):
    parent=Path(c['study_root']);old=Path(c['original_router']);plan=read(old/'resolved_configuration.json');root=parent/ds/'full'
    count=parent/'teacher'/ds/'counts.json';archive=read(count)
    for row in archive['samples'].values():
        if sha(row['evidence_path'])!=row['evidence_sha256']:raise ValueError('Changed teacher evidence')
    cfg=dict(plan['config']);cfg['scientific_choices']=c['scientific_choices'];cfg['replacement_fallback_counts']={ds:dict(path=str(count),sha256=sha(count))}
    cfg['pinned_inputs']=dict(cfg['pinned_inputs'],**{str(count):sha(count)})
    plan=dict(plan,config=cfg,records={ds:plan['records'][ds]},partitions={ds:plan['partitions'][ds]},populations={ds:plan['populations'][ds]},
        candidates={ds:plan['candidates'][ds]},actual_new_deepsolo_calls=900 if ds=='icdar2015' else 0,
        scope='User-authorized retraining against qualified Generic / TRAIN-only Weak teacher, unchanged experts and independent heldout calibration',
        parent_configuration_sha256=sha(Path('configs/approximate/router_lexicon_retraining_v1.json')),
        classified_limitations=c['scientific_choices'])
    prior=read(old/'unet_selection.json');trials={ds:prior['trials'][ds]};choices={ds:prior['choices'][ds]}
    if ds=='icdar2015':
        screen=read(parent/'candidate_selection_initial.json');trials[ds]=screen['metrics'];plan['candidates'][ds]=list(screen['metrics'])
        choices[ds]=max(plan['candidates'][ds],key=lambda n:(trials[ds][n]['f1'],-plan['candidates'][ds].index(n)))
        if choices[ds] not in read(old/'unet_selection.json')['trials'][ds]:raise ValueError('New winner requires qualified prediction/map generation before training')
    code={str(p):sha(p) for folder in ['models','recognition','localize','scripts','segmentation','deepsolo','caption','scoring','pipeline','research'] for p in Path(folder).rglob('*.py')}
    plan['code_sha256']=code
    for name,digest in code.items():
        dest=root/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if sha(name)!=digest:raise ValueError('Scientific source changed during archive')
        shutil.copyfile(name,dest)
    shutil.copytree('configs',root/'code_snapshot/configs',dirs_exist_ok=True)
    (root/'experts').symlink_to(old/'experts',target_is_directory=True)
    seal(root/'resolved_configuration.json',plan)
    seal(root/'unet_selection.json',dict(choices=choices,trials=trials,selection_ids={ds:plan['partitions'][ds]['selection']},test_used=False))
    print('PREPARED',ds,choices[ds],root,flush=True)


def audit_teacher(c,ds):
    folder=Path(c['study_root'])/'teacher'/ds;data=read(folder/'counts.json');checks=0
    for sid,row in data['samples'].items():
        if sha(row['evidence_path'])!=row['evidence_sha256']:raise ValueError('Changed teacher evidence')
        evidence=read(row['evidence_path'])
        assert independent(evidence['predictions'],evidence['ground_truth'])==row['counts'];checks+=1
    if ds=='icdar2015':
        old=read(Path(c['original_router'])/'resolved_configuration.json');words=set()
        for r in old['records'][ds]['train']:
            gt,_=annotation(r);words.update(g['text'].strip().upper() for g in gt)
        assert Path(data['train_vocabulary']).read_text().splitlines()==sorted(words)
    seal(folder/'independent_audit.json',dict(status='PASS',word_recounts=checks,counts_sha256=sha(folder/'counts.json'),code_sha256=sha(__file__),test_vocabulary_used_for_training=False))


def audit_selection(c,ds):
    if ds!='icdar2015':raise ValueError('Additional expert comparison is IC15-only')
    root=Path(c['study_root']);old=Path(c['original_router']);plan=read(old/'resolved_configuration.json')
    screen=read(root/'candidate_selection_initial.json');qualification=read(root/'candidate_qualification.json')
    records={r['sample_id']:r for r in plan['records'][ds]['train']};ids=plan['partitions'][ds]['selection']
    assert screen['selection_ids']==ids and not screen['test_used']
    from segmentation.icdar2015_target70 import emitted_predictions
    branches={}
    for name,expected in screen['metrics'].items():
        if name in screen['additional']:
            source=screen['additional'][name];q=qualification['candidates'][name]
            for p,h in [(q['policy_report'],q['policy_sha256']),(q['checkpoint'],q['checkpoint_sha256']),
                        (source['per_image'],q['per_image_sha256'])]:assert sha(p)==h
            report=read(Path(q['training_run'])/'run_manifest.json');assert report['training_complete'] and report['status']=='SUCCESS'
            raw={r['sample_id'].split(':')[-1]:r for r in read(source['per_image'])}
        counts={}
        for sid in ids:
            gt,validity=annotation(records[sid])
            if name in screen['additional']:
                pp=emitted_predictions(raw[sid.split(':')[-1]]['predictions'],source['policy']['emission_policy'])
                pp=[dict(p,raw_text=p['text']) for p in pp]
            else:pp=read(old/'experts'/name/(key(sid)+'.json'))['rows']
            counts[sid]=independent(predictions(pp,records[sid],validity),gt)
        actual=aggregate(counts.values());assert actual==expected;branches[name]=counts
    tt=read(root/'totaltext_ic15_test.json')
    for sid,v in tt['per_image'].items():assert independent(v['predictions'],v['ground_truth'])==v['counts']
    assert aggregate(v['counts'] for v in tt['per_image'].values())['f1']==tt['metrics']['f1']
    seal(root/'candidate_selection_audit.json',dict(status='PASS',selection_images=len(ids),candidates=len(branches),
        word_recounts=len(branches)*len(ids),totaltext_test_images=len(tt['per_image']),
        selection_sha256=sha(root/'candidate_selection_initial.json'),qualification_sha256=sha(root/'candidate_qualification.json'),
        totaltext_test_sha256=sha(root/'totaltext_ic15_test.json'),branches=branches,test_used_for_selection=False,code_sha256=sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--dataset',choices=['icdar13','icdar2015'],required=True);p.add_argument('--stage',choices=['teacher','prepare','audit_teacher','audit_selection'],required=True);a=p.parse_args();c=checked_config(a.config)
    {'teacher':teacher,'prepare':prepare,'audit_teacher':audit_teacher,'audit_selection':audit_selection}[a.stage](c,a.dataset)
