"""Independent official-CSV recount and exact61-image decision audit."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def norm(s):return ''.join(c for c in s.casefold() if c.isalnum())


def audit(root):
    plan=read(root/'resolved_configuration.json');result=read(root/'results.json');data=read(root/'test_scores.json')
    c=plan['config'];parts=plan['partitions'];frozen=read(root/'frozen_router.json')
    if plan['mode']!='approximate' or result['test_used_for_selection'] or not result['new_router_training']:
        raise ValueError('Expected newly trained approximate router without TEST selection')
    if any(set(parts['test'])&set(parts[k]) for k in ('fit','selection','calibration')):
        raise ValueError('TEST entered the training populations')
    if frozen['fit_ids']!=parts['fit'] or frozen['selection_ids']!=parts['selection'] or frozen['calibration_ids']!=parts['calibration']:
        raise ValueError('Recorded fitting/calibration scope changed')
    teacher=read(c['teacher_counts']);records={r['sample_id']:r for r in plan['records']};branches={};pins={}
    for sid in parts['test']:
        r=records[sid];path=Path(r['annotation_path'])
        if sha(path)!=r['annotation_sha256'] or sha(r['image_path'])!=r['image_sha256']:
            raise ValueError('Official annotation/RGB identity changed')
        with path.open(encoding='utf-8-sig',newline='') as f:labels=list(csv.reader(f,skipinitialspace=True))
        if any(len(row)!=5 for row in labels):raise ValueError('Expected official Task1 CSV')
        truth=Counter(norm(row[4]) for row in labels if norm(row[4]));branches[sid]={}
        light_path=root/'experts'/(sid.replace(':','__')+'.json');light=read(light_path)
        t=teacher['samples'][sid];deep=read(t['evidence_path'])
        if sha(t['evidence_path'])!=t['evidence_sha256']:raise ValueError('Teacher evidence changed')
        for name,rows in [('light',light['rows']),('deepsolo',[p for p in deep['predictions'] if not p['ignored']])]:
            strings=[row['raw_text'] if name=='light' else row['text'] for row in rows]
            observed=Counter(norm(s) for s in strings if norm(s));tp=sum((truth&observed).values())
            value=dict(tp=tp,fp=len(rows)-tp,fn=len(labels)-tp,detected_words=len(rows),gt_words=len(labels),all_outputs=len(rows),ignored_outputs=0)
            if value!=data['branches'][sid][name]:raise ValueError('Independent CSV word recount differs: '+sid+':'+name)
            branches[sid][name]=value
        pins[str(light_path)]=sha(light_path);pins[t['evidence_path']]=sha(t['evidence_path'])
    scores=data['scores'];keys=data['tie_keys'];order=sorted((s for s in scores if scores[s] is not None),key=lambda s:(-scores[s],keys[s],s))
    requested=round(len(parts['test'])*c['primary_deepsolo_images']/c['expected_populations']['test'])
    accepted=set(order[:len(parts['test'])-requested])
    if accepted!=set(result['primary']['accepted_ids']) or len(parts['test'])-len(accepted)!=requested:
        raise ValueError('Independent exact61 rank decision differs')
    for name in ['primary','all_light','all_deepsolo','frozen_old_gate_control','conditional_gt_oracle']:
        point=result[name];keep=set(point['accepted_ids']);counts={k:sum(branches[s]['light' if s in keep else 'deepsolo'][k] for s in branches) for k in next(iter(branches.values()))['light']}
        if any(counts[k]!=point['combined'][k] for k in counts):raise ValueError('Independent route sum failed: '+name)
        if 2*counts['tp']/(counts['detected_words']+counts['gt_words'])!=point['combined']['f1']:raise ValueError('Independent route F1 differs')
    selected=[dict(sample_id=sid,image_path=records[sid]['image_path'],rgb_sha256=keys[sid],router_score=scores[sid]) for sid in sorted(set(parts['test'])-accepted)]
    with (root/'deepsolo_selected_images.json').open('x') as f:json.dump(dict(deepsolo_images=len(selected),total_images=len(parts['test']),deepsolo_percentage=100*len(selected)/len(parts['test']),images=selected),f,indent=2);f.write('\n')
    with (root/'deepsolo_selected_images.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    value=dict(status='PASS',mode='approximate',claim_label='approximate reproduction',official_csv_images=len(branches),expert_recounts=2*len(branches),independent_route_recounts=5,
        deepsolo_images=requested,total_images=len(branches),deepsolo_percentage=100*requested/len(branches),results_sha256=sha(root/'results.json'),
        model_sha256=sha(root/result['winner']['model_path']),auditor_sha256=sha(__file__),per_image_sha256=pins)
    with (root/'independent_csv_audit.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k!='per_image_sha256'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-root',type=Path,required=True);args=p.parse_args();audit(args.run_root)
