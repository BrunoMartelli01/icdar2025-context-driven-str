"""Compare a shared reread grid on fixed native validation routes only."""

import argparse
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts.run_uniform_unet_training import read,write
from scripts.train_uniform_word_union import data
from scripts.audit_uniform_risk_fusion import count
from recognition.uniform_word_union import select
from recognition.uniform_risk_fusion import combine
from recognition.uniform_lexical_risk import lexicon
from recognition.fallback_reread import reread_rows
from recognition.uniform_utility_fusion import paired_f1_lower
from segmentation.uniform_unet_fusion import file_hash,aggregate_counts


def run(path):
    c=read(path)
    if c['mode']!='approximate' or path.resolve().parent.name!='approximate':raise ValueError('Explicit approximate validation diagnostic required')
    for p,h in c['pinned_inputs'].items():assert file_hash(p)==h
    reader=Path(c['reader_root']);assert read(reader/'independent_audit.json')['status']=='PASS'
    base=read(c['base_pipeline_config']);domains=data(base);parent=Path(base['output_root']);model=joblib.load(parent/'pooled_union_model.joblib')
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);write(root/'resolved_configuration.json',c);results={}
    for ds,v in domains.items():
        ids=v['plan']['partitions']['selection'];scenes=v['scenes'];frozen=read(parent/ds/'selection_frozen.json')
        idx=read(reader/ds/'index.json');reads={}
        for sid in ids:
            e=idx[sid];assert file_hash(e['path'])==e['sha256'];sample=read(e['path']);assert not sample['ground_truth_used']
            reads[sid]={str(r['instance_id']):r['rereadings'] for r in sample['rows']}
        probabilities={s:model.predict_proba(scenes[s]['x'])[:,1] if len(scenes[s]['x']) else [] for s in ids}
        dictionary=lexicon(base['lexicons'][ds],'train');indices=np.random.default_rng(c['seed']).integers(len(ids),size=(c['bootstrap_draws'],len(ids)));results[ds]={}
        for label,quotas in c['reference_controls'].items():
            q=str(quotas[ds]);gate=frozen['policies'][q];accepted=set(gate['accepted_validation_ids']);baseline={}
            for sid in ids:
                s=scenes[sid];light=[r for r in s['light'] if r['word_probability']>=gate['admission_threshold']]
                if sid in accepted:rows=light
                elif gate['learned_union_threshold'] is not None:rows=select(s['rows'],probabilities[sid],gate['learned_union_threshold'],base['learned_word_union']['conflict_iou'])
                else:rows=combine(dict(rows=light,probabilities=s['probabilities'],trace=s['trace']),s['heavy'],gate['fusion'],gate['complement'])
                baseline[sid]=rows
            def recount(values):
                counts=[]
                for sid,rows in values.items():
                    s=scenes[sid];flags=[s['flags'][int(r['source_row'])] if sid in accepted or r.get('fixed_light_complement') else r['ignored'] for r in rows]
                    counts.append(count(rows,s['truth'],flags))
                return aggregate_counts(counts),np.array([[r[k] for k in ['tp','fp','fn']] for r in counts])
            original,triples=recount(baseline);assert original==gate['counts'];trials=[]
            for padding in c['paddings']:
                for threshold in c['confidence_thresholds']:
                    policy=dict(enabled=True,padding=padding,threshold=threshold,confidence_kind='sequence',require_lexical_match=False)
                    proposed={sid:rows if sid in accepted else reread_rows(rows,reads[sid],dictionary,policy) for sid,rows in baseline.items()}
                    aggregate,other=recount(proposed);lower=paired_f1_lower(triples,other,indices,c['bootstrap_alpha'])
                    trials.append(dict(policy=policy,counts=aggregate,paired_lower_gain=lower,eligible=lower>0.))
            results[ds][label]=dict(baseline=original,trials=trials,deepsolo_images=len(ids)-len(accepted),total_images=len(ids),
                deepsolo_percentage=100*(len(ids)-len(accepted))/len(ids),reference_full_dataset_quota=int(q),actual_new_neural_calls=0)
        write(root/(ds+'.json'),results[ds])
        print(ds,[(k,round(v['baseline']['f1'],4),round(max(t['counts']['f1'] for t in v['trials']),4),sum(t['eligible'] for t in v['trials'])) for k,v in results[ds].items()],flush=True)
    write(root/'complete.json',dict(status='COMPLETE',mode='approximate',claim_label='approximate reproduction',test_used=False,
        scope='Native validation diagnostic on unchanged primary/common80/common90 routes; no TEST or deployment selection claim',source_sha256=file_hash(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=2):run(a.config)
