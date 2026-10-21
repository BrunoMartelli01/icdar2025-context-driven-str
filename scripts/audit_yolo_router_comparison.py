"""Independent count/selection replay and readable report, without neural inference."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import unicodedata

REPO=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def seal_valid(value):
    content={k:v for k,v in value.items() if k!='payload_sha256'}
    return hashlib.sha256(json.dumps(content,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()==value['payload_sha256']


def word_counts(predictions, gt):
    care=[p for p in predictions if not p['ignored']]
    bag=Counter(p['text'].casefold() for p in care if p['text'] and p.get('invalid_envelope') is None)
    truth=Counter(g['text'].casefold() for g in gt if g['text'])
    tp=sum((bag & truth).values())
    return {'tp':tp,'fp':len(care)-tp,'fn':len(gt)-tp,'detected_words':len(care),
            'gt_words':len(gt),'all_outputs':len(predictions),'ignored_outputs':len(predictions)-len(care)}


def normalized(text):
    return ''.join(c for c in text.casefold() if unicodedata.category(c)[0] not in ('P','S'))


def independent_joint(r):
    if not r['token_probabilities'] or not normalized(r['raw_text']):return 0.
    geometric=math.exp(sum(r['token_log_probabilities'])/len(r['token_log_probabilities']))
    quality=math.sqrt(geometric*min(r['token_probabilities']))
    return math.sqrt(r['detector_score']*quality)


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def near(a,b):
    if a is None or b is None:assert a is b
    else:assert math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-12),(a,b)


def audit(root):
    p=read(root/'resolved_configuration.json');r=read(root/'results.json');c=read(root/'frozen_train_thresholds.json')
    assert r['plan_sha256']==sha(root/'resolved_configuration.json')
    assert r['calibration_sha256']==sha(root/'frozen_train_thresholds.json')
    assert seal_valid(c) and c['test_features_present_at_freeze']==0 and not c['test_labels_used']
    assert not r['test_tuning'] and not p['config']['table_5_claim']
    inputs={}
    for filename,digest in p['source_input_sha256'].items():
        assert sha(filename)==digest;inputs[filename]=digest
    for filename,digest in p['code_sha256'].items():
        assert sha(REPO/filename)==digest
        target=root/'code_snapshot'/filename;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REPO/filename,target)
        assert sha(target)==digest
    previous=Path(p['paths']['source_root'])
    train_report=read(previous/'train/results.json')
    old_ds={x['sample_id']:x['predictions_all'] for x in train_report['per_image'] if x['action']=='deepsolo'}
    test_audit=read(previous/'test_independent_audit.json')
    old_ds.update({x['sample_id']:x['predictions'] for x in test_audit['deepsolo_only_per_image']})
    completion_path=Path(p['paths']['completed_branches']).parent/'completion.json'
    completed=read(completion_path)
    assert completed['word_branch_sha256']==sha(p['paths']['completed_branches'])
    old_ds.update({sid:x['predictions'] for sid,x in completed['per_image'].items()})
    for file in (previous/'train/results.json',previous/'test_independent_audit.json',completion_path):inputs[str(file)]=sha(file)
    b=read(root/'word_branch_metrics.json')['splits'];feature_count=crop_count=point_count=0
    features={}
    peaks={'clip':{'peak_allocated_bytes':0,'peak_reserved_bytes':0},'yolo':{'peak_allocated_bytes':0,'peak_reserved_bytes':0}}
    for split,records in p['samples'].items():
        assert len(records)==len(b[split])==r['splits'][split]['images']
        assert sum(x['yolo']['gt_words'] for x in b[split].values())==r['splits'][split]['gt_words']
        for s in records:
            sid=s['sample_id'];fpath=root/split/'features'/(sid.replace(':','__')+'.json');f=read(fpath)
            assert seal_valid(f) and f['plan_sha256']==r['plan_sha256'] and f['sample_id']==sid
            assert f['image_sha256']==s['image_sha256']==sha(s['image_path'])
            assert not f['ground_truth_in_scoring'];feature_count+=1
            features[sid]=f
            if f['paper_decision'] is not None:
                from rapidfuzz.fuzz import ratio
                decision=f['paper_decision'];pc=p['config']['paper_router']
                assert decision['semantic_winner']==('T1' if decision['S1']>decision['S3'] else 'T3')
                for index in ('1','3'):
                    lexical=ratio(decision['T'+index],s['caption']['text'])/100
                    near(lexical,decision['L'+index])
                    near(pc['alpha']*decision['S'+index]+pc['beta']*lexical,decision['C'+index])
                near(decision['C'],decision['C1'] if decision['semantic_winner']=='T1' else decision['C3'])
            branch=b[split][sid]
            for engine,preds in [('yolo',branch['yolo_predictions']),('deepsolo',old_ds[sid]),
                                 ('t1',[{'text':f['t1']['text'],'ignored':False}])]:
                counted=word_counts(preds,branch['ground_truth'])
                assert all(counted[k]==branch[engine][k] for k in counted),(sid,engine)
            raw=(read(root/'train/ocr'/(sid.replace(':','__')+'.json'))['rows'] if split=='train' else
                 read(Path(p['paths']['yolo_test_root'])/'ocr'/(sid.replace(':','__')+'.json'))['rows'])
            selected=[]
            for row in sorted(raw,key=lambda v:(-independent_joint(v),v['candidate_id'])):
                if independent_joint(row)<p['config']['selection']['joint_threshold'] or not normalized(row['raw_text']):continue
                if any(normalized(row['raw_text'])==normalized(x['raw_text']) and
                       iou(row['bbox'],x['bbox'])>p['config']['selection']['nms_iou'] for x in selected):continue
                selected.append(row)
            assert [v['candidate_id'] for v in selected]==[v['candidate_id'] for v in f['rows']]
            from PIL import Image
            import numpy as np
            with Image.open(s['image_path']) as im:rgb=np.asarray(im.convert('RGB'))
            for row in f['rows']:
                x1,y1,x2,y2=row['bbox'];pixels=np.ascontiguousarray(rgb[y1:y2,x1:x2])
                assert hashlib.sha256(pixels.tobytes()).hexdigest()==row['crop_rgb_sha256'];crop_count+=1
            expected=None
            if f['rows'] and all(v['raw_text'].strip() and v['clip_word_advantage'] is not None for v in f['rows']):
                def rank(ref,v):return (bisect_left(ref,v)+bisect_right(ref,v))/(2*len(ref))
                expected=min(math.sqrt(rank(c['reference']['native'],v['confidence'])*
                    rank(c['reference']['word_advantage'],v['clip_word_advantage'])) for v in f['rows'])
            near(expected,r['splits'][split]['scores']['clip'][sid])
            for k in peaks['clip']:peaks['clip'][k]=max(peaks['clip'][k],f['clip_peak'][k] or 0)
            if split=='train':
                shard=read(root/'train/ocr'/(sid.replace(':','__')+'.json'))
                assert shard['yolo_input_shapes'] and seal_valid(shard)
                for k in peaks['yolo']:peaks['yolo'][k]=max(peaks['yolo'][k],shard['detector_peak'][k] or 0)
        ids=list(b[split])
        for label,point in r['splits'][split]['points'].items():
            accepted=set(point['accepted_ids'])
            if '/train_coverage_' in label:
                name,fraction=label.split('/train_coverage_');cut=c['cuts'][name][fraction]
                assert point['threshold']==cut
                values=r['splits'][split]['scores'][name]
                assert accepted=={sid for sid in ids if cut is not None and values[sid] is not None and values[sid]>=cut}
            if '/same_budget_' in label:
                name,count=label.split('/same_budget_');values=r['splits'][split]['scores'][name]
                ranked=sorted((sid for sid in ids if values[sid] is not None),key=lambda sid:(-values[sid],sid))
                assert accepted==set(ranked[:int(count)])
            counts=Counter()
            for sid,engine in point['sources'].items():
                assert engine in (('yolo','t1') if sid in accepted else ('deepsolo',))
                expected_engine=('t1' if label.startswith('paper_selected/') and features[sid]['paper_decision'] is not None
                                 and features[sid]['paper_decision']['semantic_winner']=='T1' else 'yolo')
                assert engine==(expected_engine if sid in accepted else 'deepsolo')
                counts.update({k:b[split][sid][engine][k] for k in ('tp','fp','fn','detected_words','gt_words','all_outputs','ignored_outputs')})
            assert all(counts[k]==point['combined'][k] for k in counts)
            near(counts['tp']/sum(counts[k] for k in ('tp','fp','fn')),point['combined']['accuracy_tn0'])
            near(2*counts['tp']/(2*counts['tp']+counts['fp']+counts['fn']),point['combined']['f1'])
            point_count+=1
    # Additional predeclared confidence controls: no new gates, only inference-free intervals.
    from recognition.yolo_router_comparison import paired_bootstrap
    supplemental={}
    for split,data in r['splits'].items():
        budget=round(data['images']*.5);left=data['points'][f'clip/same_budget_{budget}']
        for label in ('yolo_only','deepsolo_only',f'native/same_budget_{budget}',f'joint/same_budget_{budget}'):
            supplemental[split+'/clip_vs_'+label]=paired_bootstrap(b[split],left,data['points'][label],
                p['config']['bootstrap_iterations'],p['config']['execution']['seed'])
    (root/'supplemental_bootstrap.json').write_text(json.dumps(supplemental,indent=2,sort_keys=True)+'\n')
    result={'status':'PASS','features':feature_count,'verified_rgb_crops':crop_count,
        'independently_replayed_points':point_count,'independent_fallback_word_replays':feature_count,
        'memory':peaks,'results_sha256':sha(root/'results.json'),'plan_sha256':r['plan_sha256'],
        'input_sha256':inputs,'audit_script_sha256':sha(Path(__file__)),
        'supplemental_bootstrap_sha256':sha(root/'supplemental_bootstrap.json'),
        'failures':[],'prior_failed_smoke':'smoke/failure.json: Namespace.output_root missing; fixed in smoke-v2; no failed sample omitted',
        'prior_test_collection_failure':'duplicate test module names; runner tests renamed; final67 tests pass'}
    (root/'independent_audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'}))
    return result


def report(root, render):
    r=read(root/'results.json');p=read(root/'resolved_configuration.json')
    pct=lambda v:'—' if v is None else f'{100*v:.2f}%'
    lines=['# YOLO: confronto appaiato dei router','',
        '**Approximate reproduction; confronto esplorativo, non replica STRICT.**','',
        'Stesse immagini, stesse parole YOLO640+PARSeq (joint0.60/NMS0.5), stesso DeepSolo R50.',
        'TRAIN125 (983 parole) e TEST300 (2204 parole) restano separati. CDF e soglie fit solo su TRAIN80;',
        'il gate CLIP B16 e i prompt sono fissati prima del nuovo esperimento, senza scelta della famiglia su TEST.',
        'Il detector/selelettore YOLO era già stato esplorato su TEST: non è una nuova validazione indipendente.','',
        'Metrica primaria: multiset di parole per immagine, casefold esatto, punteggiatura mantenuta, senza IoU.',
        'P=TP/(TP+FP); TN0=TP/(TP+FP+FN); F1=2TP/(2TP+FP+FN). Le parole mancate restano FN.',
        'Ignore solo in valutazione: supporto nativo256 interamente invalido, rettangolo YOLO / poligono DeepSolo.',
        'Questa regola differisce dal precedente report YOLO con neutralizzazione polygon-overlap>0.5.','']
    labels=['yolo_only','deepsolo_only','paper_selected/paper_tau','clip/train_coverage_0.5',
            'paper_selected/train_coverage_0.5','native/train_coverage_0.5','joint/train_coverage_0.5']
    for split,data in r['splits'].items():
        lines += [f'## {split.upper()} — soglie trasferite dal TRAIN','',
            '| Metodo | YOLO/T1 accettate | DeepSolo | Parole finali | Precisione | Recall | F1 | TN0 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for label in labels:
            point=data['points'][label];m=point['combined']
            lines.append(f"| {label} | {point['accepted_images']} | {point['fallback_images']} | {m['detected_words']} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} | {pct(m['accuracy_tn0'])} |")
        lines += ['',f'## {split.upper()} — parità di chiamate a DeepSolo','',
            'Budget descrittivi: ordinamento dei punteggi, con spareggio per ID; non sono soglie scalari trasferite.','',
            '| Accettate | CLIP P parole accettate | CLIP complete | CLIP F1 finale | Paper F1 finale | CLIP TN0 | Paper TN0 | ΔTN0 [IC95%] |',
            '|---:|---:|---:|---:|---:|---:|---:|---|']
        for frac in p['config']['coverage_targets']:
            budget=round(data['images']*frac);a=data['points'][f'clip/same_budget_{budget}'];b=data['points'][f'paper_selected/same_budget_{budget}']
            ci=r['paired_bootstrap'][f'{split}/same_budget_{budget}']
            lines.append(f"| {budget} | {pct(a['accepted_metrics']['precision'])} | {a['accepted_metrics']['complete_images']}/{a['accepted_images']} | {pct(a['combined']['f1'])} | {pct(b['combined']['f1'])} | {pct(a['combined']['accuracy_tn0'])} | {pct(b['combined']['accuracy_tn0'])} | {100*ci['estimate']:+.2f} [{100*ci['ci95'][0]:+.2f}, {100*ci['ci95'][1]:+.2f}] pp |")
        ideal=data['oracle']
        lines += ['',f"Oracolo GT YOLO/DeepSolo: F1 {pct(ideal['combined']['f1'])}, TN0 {pct(ideal['combined']['accuracy_tn0'])}, "
                  f"{ideal['accepted_images']} immagini YOLO e {ideal['fallback_images']} DeepSolo. Non è un router utilizzabile.",'']
    lines += ['## Confronto sostenibile con il paper','',
        'La Table5 riporta recognition accuracy TotalText78.4% (R50) e79.8% (ViTAEv2-S), senza definire',
        'adeguatamente denominatore, matching e ignore. I nostri F1/TN0/precision non sono quella metrica.',
        'Un eventuale superamento numerico non dimostra di aver superato il paper. Qui il confronto valido',
        'è CLIP contro la nostra baseline approssimata del router del paper con lo stesso detector/fallback.',
        'Manca anche il checkpoint DeepSolo esatto degli autori. La baseline con tau0.8 che manda tutto a',
        'DeepSolo non dimostra la qualità della selezione: confrontare sempre costo e accuracy insieme.','',
        'Gli IC sono bootstrap appaiati di5000 ricampionamenti di immagini, condizionati a gate fissi;',
        'nessuna correzione per confronti multipli. TRAIN è diagnostico: DeepSolo è stato fine-tuned su TotalTextTRAIN.','',
        '## Riproduzione','',
        '```bash','.venvs/context-str-yolo-py311/bin/python -m scripts.run_yolo_router_comparison --mode approximate','```','',
        'Configurazioni isolate: `configs/approximate/yolo_router_comparison_v1.json` e relativo local_paths.',
        'Output: `resolved_configuration.json`, `frozen_train_thresholds.json` (zero feature TEST al freeze),',
        '`word_branch_metrics.json`, `results.json`, `comparison.csv`, per-image OCR/CLIP/MPNet,',
        '`independent_audit.json` e snapshot delle implementazioni. Nessun pacchetto installato.','']
    (root/'comparison.md').write_text('\n'.join(lines))
    if render:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
        for ax,(split,data) in zip(axes,r['splits'].items()):
            for family,name in [('clip','CLIP B16'),('paper_selected','Paper (approx.)'),('native','PARSeq confidence'),('joint','YOLO/PARSeq confidence')]:
                xs=[];ys=[]
                for frac in p['config']['coverage_targets']:
                    budget=round(data['images']*frac);point=data['points'][f'{family}/same_budget_{budget}']
                    xs.append(100*point['accepted_images']/data['images']);ys.append(100*point['combined']['accuracy_tn0'])
                ax.plot(xs,ys,marker='o',label=name)
            for label,color in [('yolo_only','gray'),('deepsolo_only','black')]:
                ax.axhline(100*data['points'][label]['combined']['accuracy_tn0'],color=color,linestyle='--',label=label)
            ax.set_title(split.upper());ax.set_xlabel('Accepted images (%)');ax.grid(alpha=.25)
        axes[0].set_ylabel('Whole-system TN0 (%)');axes[1].legend(fontsize=8)
        fig.suptitle('Approximate reproduction — frozen YOLO/PARSeq, same fallback; descriptive budgets')
        fig.tight_layout();fig.savefig(root/'coverage_curves.png',dpi=180);fig.savefig(root/'coverage_curves.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    if not args.render_only:audit(args.root)
    report(args.root,args.render_only)


if __name__=='__main__':main()
