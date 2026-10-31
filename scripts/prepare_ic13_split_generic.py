"""TRAIN-only dataset Generic preparation; frozen official TEST resource unchanged."""

import argparse
from pathlib import Path

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.deepsolo_word_f1 import recognition_confidence
from recognition.clip_router_evaluation import deepsolo_predictions
from scripts.icdar_budget_data import annotation,read,sha,key
from scripts.run_yolo_f1_quality_study import seal
from scripts.retrain_icdar_lexicon_router import independent


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=read(a.config);root=Path(c['study_root']);plan=read(Path(c['original_router_root'])/'resolved_configuration.json');reference=read(c['teacher_reference'])
    assert c['mode']=='approximate';canonical={w.strip().upper():w.strip() for w in Path(c['train_generic_base']).read_text().splitlines()};train_words={};pins={}
    for r in plan['records']['icdar13']['train']:
        gt,_=annotation(r);pins[r['annotation_path']]=r['annotation_sha256']
        for g in gt:train_words.setdefault(g['text'].strip().upper(),g['text'].strip())
    canonical.update(train_words);words=sorted(canonical)
    if '' in canonical:raise ValueError('Empty raw dictionary entry')
    vocab=root/'train_generic_vocabulary.txt';pairs=root/'train_generic_pairs.json'
    with vocab.open('x') as f:f.write('\n'.join(words)+'\n')
    seal(pairs,canonical);lex=DeepSoloLexicon(words,canonical,distance_exclusive=1.5)
    samples={sid:v for sid,v in reference['samples'].items() if ':train:' not in sid}
    for i,r in enumerate(plan['records']['icdar13']['train'],1):
        sid=r['sample_id'];path=Path(c['teacher_raw_train'])/(key(sid)+'.json');raw=read(path);assert raw['response']['success'] and raw['contract']['image_sha256']==r['image_sha256'];pins[str(path)]=sha(path)
        neural=c['teacher_neural'];gt,valid=annotation(r);rows=[dict(instance_id=d['source_index'],text=d['text'],score=d['score']) for d in raw['response']['detections'] if d['score']>=neural['score_threshold'] and recognition_confidence(d['raw']['recognition_class_probabilities'])>=neural['recognition_threshold']]
        rows.sort(key=lambda p:(-p['score'],p['instance_id']));rows,trace=lex.correct(rows,unmatched_policy=c['train_teacher_unmatched_policy']);source={d['source_index']:d for d in raw['response']['detections']}
        detections=[dict(source[row['instance_id']],text=row['text']) for row in rows];pp=deepsolo_predictions(dict(detections=detections),r['image_wh'],r['orientation'],valid,neural['score_threshold'])
        for row in pp:
            if row['invalid_envelope'] is not None:row['text']=''
        counts=independent(pp,gt);evidence=root/'teacher/icdar13/per_image'/(key(sid)+'.json');seal(evidence,dict(sample_id=sid,predictions=pp,ground_truth=gt,counts=counts,decisions=trace,raw_path=str(path),raw_sha256=pins[str(path)]))
        samples[sid]=dict(image_sha256=r['image_sha256'],counts=counts,evidence_path=str(evidence),evidence_sha256=sha(evidence))
        if i%100==0:print('TRAIN-only Generic teacher',i,'/229',flush=True)
    seal(root/'teacher/icdar13/counts.json',dict(mode='approximate',claim_label='approximate reproduction',samples=samples,source='Common Generic base plus TRAIN-only care words; source rejection on TRAIN, official Generic keep-unmatched TEST reference unchanged',input_sha256=pins,train_vocabulary_sha256=sha(vocab),train_pairs_sha256=sha(pairs)))
    seal(root/'train_generic_provenance.json',dict(status='PASS',common_base_path=c['train_generic_base'],common_base_sha256=sha(c['train_generic_base']),train_annotations_sha256={r['annotation_path']:r['annotation_sha256'] for r in plan['records']['icdar13']['train']},words=len(words),train_word_types=len(train_words),test_transcriptions_read=0,official_test_dictionary_unchanged=True,config_sha256=sha(a.config),code_sha256=sha(__file__)))
    print('PASS TRAIN-only Generic',len(words),'entries',len(train_words),'TRAIN types',flush=True)


if __name__=='__main__':main()
