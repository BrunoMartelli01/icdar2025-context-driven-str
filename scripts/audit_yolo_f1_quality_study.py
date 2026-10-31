"""Independent rational-count audit of the offline F1-target study."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
FIELDS=('tp','fp','fn','detected_words','gt_words','all_outputs','ignored_outputs')


def f1(c):
    den=2*c['tp']+c['fp']+c['fn']
    return Fraction(2*c['tp'],den)if den else Fraction(0)


def quality(c,q):
    return bool(2*c['tp']+c['fp']+c['fn'])and f1(c)>=Fraction(str(q))


def aggregate(rows):
    rows=list(rows);c={k:sum(r[k]for r in rows)for k in FIELDS}
    assert c['tp']+c['fp']==c['detected_words']and c['tp']+c['fn']==c['gt_words']
    d=2*c['tp']+c['fp']+c['fn'];j=c['tp']+c['fp']+c['fn']
    return {**c,'images':len(rows),'f1':float(f1(c))if d else None,'accuracy_tn0':c['tp']/j if j else None,
        'precision':c['tp']/c['detected_words']if c['detected_words']else None,
        'recall':c['tp']/c['gt_words']if c['gt_words']else None}


def sealed(p,ph):
    v=read(p);expected=hashlib.sha256(json.dumps({k:x for k,x in v.items()if k!='payload_sha256'},
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert v['plan_sha256']==ph and v['payload_sha256']==expected
    return v


def point(branches,ids,accepted,q):
    a=set(accepted);true={s for s in ids if quality(branches[s]['light'],q)};g=len(a&true)
    return {'accepted_ids':sorted(a),'accepted_images':len(a),'fallback_images':len(ids)-len(a),
        'good_images':g,'bad_images':len(a)-g,'available_good_images':len(true),
        'good_fraction':g/len(a)if a else None,'good_recall':g/len(true)if true else None,
        'classification_tp':g,'classification_fp':len(a)-g,'classification_fn':len(true)-g,
        'classification_tn':len((set(ids)-a)-true),'classification_accuracy':(g+len((set(ids)-a)-true))/len(ids),
        'accepted_macro_image_f1':float(sum((f1(branches[s]['light'])for s in a),Fraction(0))/len(a))if a else None,
        'combined':aggregate(branches[s]['light'if s in a else'deepsolo']for s in ids),
        'accepted_words':aggregate(branches[s]['light']for s in a),
        'deepsolo_rejected_words':aggregate(branches[s]['deepsolo']for s in ids if s not in a)}


def check_point(actual,expected):
    for key,value in expected.items():
        if isinstance(value,dict):assert actual[key]==value,key
        elif isinstance(value,float):assert abs(actual[key]-value)<1e-12,key
        else:assert actual[key]==value,key


def curve(scores,branches,ids,qs):
    out=[];label={q:{s:quality(branches[s]['light'],q)for s in ids}for q in qs}
    for cut in sorted({scores[s]for s in ids if scores[s]is not None}):
        a={s for s in ids if scores[s]is not None and scores[s]>=cut}
        c=aggregate(branches[s]['light'if s in a else'deepsolo']for s in ids)
        out.append({'threshold':cut,'accepted_images':len(a),'counts':c,
            'good':{str(q):sum(label[q][s]for s in a)for q in qs}})
    return out


def cut_choice(curves,q,purity,minimum,objective):
    eligible=[c for c in curves if c['accepted_images']>=minimum and
        Fraction(c['good'][str(q)],c['accepted_images'])>=Fraction(str(purity))]
    def key(c):
        p=Fraction(c['good'][str(q)],c['accepted_images']);f=f1(c['counts'])
        return (c['accepted_images'],p,f,c['threshold'])if objective=='max_bypass'else(f,c['accepted_images'],p,c['threshold'])
    return max(eligible,key=key)if eligible else None


def independent_proxies(rows,consistency,context,reference,settings):
    """Rebuild the new rules without importing the scientific implementation."""
    names=('native_mean','native_lower_quantile','native_harmonic','clip_mean',
        'word_joint_mean','word_joint_lower_quantile','proxy_f1_native','proxy_f1_stable',
        'proxy_f1_visual','base_proxy_f1','scale_native_proxy','context_proxy_f1',
        'proxy_f1_duplicate_penalty')
    if not rows or any(not r['raw_text'].strip()for r in rows):return dict.fromkeys(names)
    stability={w['candidate_id']:w['minimum_edit_agreement']for w in consistency['crop_stability']['words']}
    assert set(stability)=={r['candidate_id']for r in rows}
    native=[r['confidence']for r in rows]
    ranks=[sum((x<v)+(x<=v)for x in reference)/(2*len(reference))
        for r in rows for v in [r['clip_word_advantage']]]
    crop=[stability[r['candidate_id']]for r in rows]
    joint=[(a*b*c)**(1/3)for a,b,c in zip(native,ranks,crop)]
    mean=lambda x:sum(x)/len(x)
    def quantile(values):
        ordered=sorted(values);p=(len(values)-1)*settings['lower_quantile'];i=math.floor(p)
        return ordered[i]+(p-i)*(ordered[min(i+1,len(values)-1)]-ordered[i])
    coverage=min(consistency['coverage'][k]for k in ('text_coverage','kernel_coverage'))
    proxy=lambda precision:2*precision*coverage/(1+coverage)
    duplicate=set()
    for i,row_a in enumerate(rows):
        for j,row_b in enumerate(rows):
            if i==j:continue
            a=row_a['bbox'];b=row_b['bbox']
            intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
            if union and intersection/union>=settings['duplicate_iou']:duplicate.add(i)
    native_mean=mean(native);visual=proxy(mean(joint));mpnet=context['scores']['base_mpnet']
    values=[native_mean,quantile(native),len(native)/sum(1/x for x in native)if min(native)>0 else 0.,
        mean(ranks),mean(joint),quantile(joint),proxy(native_mean),proxy(native_mean*mean(crop)),
        visual,math.sqrt(consistency['scores']['all_checks_softcrop']*visual),
        native_mean*consistency['scale_agreement']['score'],
        math.sqrt(visual*mpnet)if mpnet is not None else None,visual*(1-len(duplicate)/len(rows))]
    return dict(zip(names,values))


def ranked(scores,ids):
    return sorted((s for s in ids if scores[s]is not None),key=lambda s:(-scores[s],s))


def audit(root):
    plan=read(root/'resolved_configuration.json');cfg=plan['config'];ph=sha(root/'resolved_configuration.json')
    result=read(root/'results.json');frozen=read(root/'frozen_thresholds.json')
    assert result['status']=='SUCCESS'and result['plan_sha256']==ph and result['optimizer_steps']==0
    assert result['new_model_inference']is False and result['network_training']is False
    assert result['thresholds_sha256']==sha(root/'frozen_thresholds.json')
    assert frozen['test_features_present_at_freeze']==0 and frozen['test_labels_used']is False
    for name,d in plan['code_sha256'].items():assert sha(root/'code_snapshot'/name)==d
    for name,d in plan['input_sha256'].items():assert sha(name)==d
    for name,pins in cfg['sources'].items():
        src=Path(plan['paths']['sources'][name]);assert sha(src/'resolved_configuration.json')==pins['plan']
        assert sha(src/'results.json')==pins['results']and sha(src/pins['audit_file'])==pins['audit']
    coverage=Path(plan['paths']['sources']['coverage']);b=read(coverage/'results.json')['resolutions']['640']['branch_metrics']
    features={split:{s['sample_id']:sealed(root/split/'features'/(s['sample_id'].replace(':','__')+'.json'),ph)
        for s in samples}for split,samples in plan['samples'].items()}
    for split,group in features.items():
        for sid,v in group.items():
            assert v['ground_truth_used']is False and v['network_training']is False and v['original_words_changed']is False
            for name in ('consistency','context','vqa'):
                old=read(Path(plan['paths']['sources'][name])/split/'features'/(sid.replace(':','__')+'.json'))
                assert all(v['scores'][f]==s for f,s in old['scores'].items())
            key=sid.replace(':','__')+'.json'
            old_coverage=read(coverage/'640'/split/'features'/key)
            old_consistency=read(Path(plan['paths']['sources']['consistency'])/split/'features'/key)
            old_context=read(Path(plan['paths']['sources']['context'])/split/'features'/key)
            rebuilt=independent_proxies(old_coverage['rows'],old_consistency,old_context,
                plan['clip_reference'],cfg['new_feature_settings'])
            for name,score in rebuilt.items():
                assert v['scores'][name]is None if score is None else abs(v['scores'][name]-score)<1e-12,name
            if b[split][sid]['light']['all_outputs']==0:assert all(x is None for x in v['scores'].values())
    sel=plan['selection_ids'];cal=plan['threshold_calibration_ids']
    assert not set(sel)&set(cal)and set(sel+cal)==set(features['train'])and not set(features['train'])&set(features['test'])
    curves={name:{f:curve({s:features['train'][s]['scores'][f]for s in ids},b['train'],ids,cfg['quality_f1_thresholds'])
        for f in plan['families']}for name,ids in [('selection',sel),('calibration',cal)]}
    for key,policy in frozen['policies'].items():
        q=policy['quality'];purity=policy['purity'];objective=policy['objective'];trials=[]
        for i,family in enumerate(plan['families']):
            c=cut_choice(curves['calibration'][family],q,purity,frozen['minimum_accepted']['calibration'],objective)
            assert policy['families'][family]['threshold']==(c['threshold']if c else None)
            c=cut_choice(curves['selection'][family],q,purity,frozen['minimum_accepted']['selection'],objective)
            if c is None:trials.append((-1,-1,-1,-i,family));continue
            p=c['good'][str(q)]/c['accepted_images'];f=float(f1(c['counts']))
            trials.append((c['accepted_images'],p,f,-i,family)if objective=='max_bypass'else(f,c['accepted_images'],p,-i,family))
        assert policy['selected_family']==max(trials)[-1]
        qualified=any(t[0]>=0 for t in trials)
        assert policy['selection_qualified']==qualified
    for q in cfg['quality_f1_thresholds']:
        for fraction in cfg['bypass_fractions']:
            n=round(len(sel)*fraction);trials=[]
            for i,family in enumerate(plan['families']):
                scores={s:features['train'][s]['scores'][family]for s in sel}
                a=ranked(scores,sel)[:n]
                if len(a)!=n:continue
                p=point(b['train'],sel,a,q)
                trials.append((p['good_images'],f1(p['combined']),-i,family))
            assert frozen['budget_selected_families'][str(q)][str(fraction)]==max(trials)[-1]
    verified=0
    for split,ids,source in [('train',list(features['train']),'train'),('selection',sel,'train'),('calibration',cal,'train'),('test',list(features['test']),'test')]:
        for key,policy in result['splits'][split]['policies'].items():
            q=float(key.split(':')[0]);assert policy['selected_family']==frozen['policies'][key]['selected_family']
            for family,p in policy['families'].items():
                cut=frozen['policies'][key]['families'][family]['threshold']
                accepted={s for s in ids if cut is not None and features[source][s]['scores'][family]is not None and
                    features[source][s]['scores'][family]>=cut}
                check_point(p,point(b[source],ids,accepted,q));verified+=1
            fit=frozen['policies'][key]
            assert policy['selection_qualified']==fit['selection_qualified']
            if fit['selection_qualified']:
                assert policy['selected']==policy['families'][policy['selected_family']]
            else:
                check_point(policy['selected'],point(b[source],ids,set(),q))
                assert policy['selected']['calibration_status']=='ABSTAIN_ALL_NO_QUALIFYING_SELECTION_FAMILY'
        for q,p in result['splits'][split]['quality_oracles'].items():
            a={s for s in ids if quality(b[source][s]['light'],float(q))}
            assert p['uses_ground_truth']is True;check_point(p,point(b[source],ids,a,float(q)));verified+=1
        assert result['splits'][split]['deepsolo_only']==aggregate(b[source][s]['deepsolo']for s in ids)
        assert result['splits'][split]['light_only']==aggregate(b[source][s]['light']for s in ids)
        for q,budgets in result['splits'][split]['budgets'].items():
            for fraction,p in budgets.items():
                family=frozen['budget_selected_families'][q][fraction]
                assert p['selected_family']==family
                n=round(len(ids)*float(fraction));assert p['requested_images']==n
                scores={s:features[source][s]['scores'][family]for s in ids}
                accepted=ranked(scores,ids)[:n]
                assert p['unfilled_budget']==n-len(accepted)
                check_point(p,point(b[source],ids,accepted,float(q)));verified+=1
    assert(root/'frozen_thresholds.json').stat().st_mtime_ns<min(p.stat().st_mtime_ns for p in(root/'test/features').glob('*.json'))
    if plan['result_class']!='SMOKE_TEST':
        assert len(features['train'])==125 and len(features['test'])==300
        assert result['splits']['train']['light_only']['gt_words']==983 and result['splits']['test']['light_only']['gt_words']==2204
    out={'status':'PASS','reproduction_mode':'approximate','claim_label':'approximate reproduction',
        'plan_sha256':ph,'results_sha256':sha(root/'results.json'),'audit_code_sha256':sha(__file__),
        'features_verified':sum(len(v)for v in features.values()),'routing_and_oracle_points_verified':verified,
        'new_feature_formulas_independently_verified':True,'fixed_budget_selection_and_points_verified':True,
        'train_only_thresholds_and_family_selection':True,'optimizer_steps':0,'new_model_inference':False,
        'word_denominators_preserved':True}
    (root/'independent_audit.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
