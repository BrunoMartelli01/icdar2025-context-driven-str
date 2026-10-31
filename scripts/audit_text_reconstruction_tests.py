"""Independently recount saved reconstruction results and validate provenance."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from recognition.scene_text_datasets import parse_totaltext_annotation


def read(path):return json.loads(Path(path).read_text())


def file_hash(path):
    result=sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):result.update(chunk)
    return result.hexdigest()


def observed_tokens(groups):
    values=[]
    for group in groups:
        for text in group['texts']:
            values.extend(text.casefold().split() or [''])
    return Counter(values)


def independent_iou(left,right):
    width=max(0,min(left[2],right[2])-max(left[0],right[0]))
    height=max(0,min(left[3],right[3])-max(left[1],right[1]))
    intersection=width*height
    denominator=(left[2]-left[0])*(left[3]-left[1])+(right[2]-right[0])*(right[3]-right[1])-intersection
    return intersection/denominator if denominator else 0.


def maximum_matches(groups,words):
    # DFS augmenting paths provide an implementation independent of primary SciPy assignment.
    predictions=[(g['bbox'],text.casefold()) for g in groups for text in g['texts']]
    adjacency=[[j for j,w in enumerate(words) if text==w['text'].casefold() and independent_iou(box,w['bbox'])>=.5] for box,text in predictions]
    owner={}
    def augment(i,seen):
        for j in adjacency[i]:
            if j in seen:continue
            seen.add(j)
            if j not in owner or augment(owner[j],seen):owner[j]=i;return True
        return False
    for i in range(len(predictions)):augment(i,set())
    return len(owner)


def covered_letters(original,candidate):
    old=''.join(original.casefold().split());new=''.join(candidate.casefold().split())
    position=0
    for char in new:
        if position<len(old) and char==old[position]:position+=1
    return position==len(old)


def load_words(sample):
    if file_hash(sample['annotation_path'])!=sample['annotation_sha256']:raise RuntimeError('Annotation changed')
    parsed=[label for label in parse_totaltext_annotation(Path(sample['annotation_path'])) if not label.ignore]
    h,w=sample['image_hw'];words=[]
    for label in parsed:
        a,b,c,d=label.bbox_xyxy
        words.append(dict(text=label.transcription,bbox=[max(0,a),max(0,b),min(w,c+1),min(h,d+1)]))
    return words


def audit(root):
    protocol=read(root/'protocol.json');cfg=protocol['resolved_config'];source=Path(cfg['paths']['source_run'])
    checks=Counter()
    for name,value in cfg['source_sha256'].items():
        if file_hash(source/name)!=value:raise RuntimeError('Original source changed: '+name)
        checks['original_run_artifacts']+=1
    for path,value in protocol['inputs_sha256'].items():
        if file_hash(path)!=value:raise RuntimeError('Original OCR export changed')
        checks['original_exports']+=1
    observations=read(root/'observations.json')
    if file_hash(root/'observations.json')!=protocol['observations_sha256']:raise RuntimeError('Observations changed')
    for section,path_key in [('spell','spell_snapshot'),('denoiser','byt5_snapshot')]:
        for name,value in cfg[section]['files_sha256'].items():
            if file_hash(Path(cfg['paths'][path_key])/name)!=value:raise RuntimeError('Pretrained artifact changed')
            checks['pretrained_files']+=1
    selection=read(root/'selection_frozen_before_evaluation.json')
    if file_hash(root/'denoiser_selected.pt')!=selection['denoiser_checkpoint_sha256']:raise RuntimeError('Selected checkpoint changed')
    history=read(root/'denoiser_history.json')
    expected_epochs=1 if protocol['smoke'] else cfg['denoiser']['epochs']
    if len(history['history'])!=expected_epochs:raise RuntimeError('Training budget incomplete')
    if sum(row['updates'] for row in history['history'])!=expected_epochs*((history['history'][0]['pairs']+cfg['denoiser']['batch_size']-1)//cfg['denoiser']['batch_size']):
        raise RuntimeError('Optimizer update budget mismatch')
    complete=read(root/'completion.json');denoised=read(root/'denoiser_calibration.json')
    calibration_data={s['sample_id']:load_words(s) for s in observations['calibration']}
    for name,choice in selection['choices'].items():
        counts=Counter();base_correct=0
        for sample in observations['calibration']:
            sid=sample['sample_id'];expected=Counter(w['text'].casefold() for w in calibration_data[sid])
            base=Counter(t for b in sample['blocks'] for t in (b['text'].casefold().split() or [''])) & expected
            groups=complete['calibration'][sid][name] if name.startswith('neuspell') else denoised[sid][name]
            predicted=observed_tokens(groups);matched=predicted & expected
            counts.update(correct=sum(matched.values()),predictions=sum(predicted.values()),gt_words=sum(expected.values()),
                          recovered=sum((matched-base).values()),damaged=sum((base-matched).values()))
            base_correct+=sum(base.values())
        if any(choice['calibration'][key]!=value for key,value in counts.items()):raise RuntimeError('Calibration summary disagrees')
        if choice['enabled']!=(counts['correct']>base_correct and counts['damaged']<=cfg['selector']['calibration_damage_limit']):
            raise RuntimeError('Policy enable/KEEP choice disagrees with calibration')
        checks['calibrated_policy_choices']+=1
    result=read(root/'results.json');summaries={}
    for phase in ['evaluation','test']:
        rows=read(root/(phase+'_per_image.json'));samples={s['sample_id']:s for s in observations[phase]}
        if set(samples)!={r['sample_id'] for r in rows} or len(samples)!=len(rows):raise RuntimeError('Missing/duplicate images')
        aggregates={}
        for row in rows:
            sample=samples[row['sample_id']]
            words=load_words(sample)
            expected=Counter(w['text'].casefold() for w in words)
            baseline=observed_tokens(row['arms']['baseline']['groups']) & expected
            parent_ids=sorted(b['id'] for b in sample['blocks'])
            for arm,data in row['arms'].items():
                groups=data['groups'];consumed=sorted(p for group in groups for p in group['parents'])
                if consumed!=parent_ids:raise RuntimeError('Input omitted/reused: '+arm)
                predicted=observed_tokens(groups);matched=predicted & expected
                recovered,damaged=sum((matched-baseline).values()),sum((baseline-matched).values())
                actual=dict(correct=sum(matched.values()),predictions=sum(predicted.values()),gt_words=sum(expected.values()),recovered=recovered,damaged=damaged)
                if any(data[k]!=value for k,value in actual.items()):raise RuntimeError('Per-image metric mismatch: '+arm)
                if actual['correct']!=sum(baseline.values())+recovered-damaged:raise RuntimeError('Gain/loss accounting mismatch')
                if 'fixed_bbox_exact' in data:
                    count=maximum_matches(groups,words)
                    if count!=data['fixed_bbox_exact']:raise RuntimeError('Independent fixed-box matcher disagrees')
                    actual['fixed_bbox_exact']=count;checks['fixed_bbox_matching']+=1
                if arm in ['denoiser_preserve','structure_denoiser_preserve']:
                    raw_name=arm.removesuffix('_preserve')
                    raw=row['arms'][raw_name]['groups']
                    for group,other in zip(groups,raw):
                        if group['parents']!=other['parents']:raise RuntimeError('Preservation policy changed grouping')
                        # For calibrated structure=KEEP, source and original text are equal;
                        # for an active structural policy, use its selected output as the reference.
                        base_arm='structure' if arm.startswith('structure_') else 'baseline'
                        refs={tuple(g['parents']):g for g in row['arms'][base_arm]['groups']}
                        reference=refs[tuple(group['parents'])]
                        if group['texts']!=reference['texts'] and not covered_letters(' '.join(reference['texts']),' '.join(group['texts'])):
                            raise RuntimeError('Letter preservation violated')
                totals=aggregates.setdefault(arm,Counter());totals.update(actual);checks['image_arm_counts']+=1
            oracle=next(r for r in result['cohorts'][phase]['bounded_structural_pool_oracle']['images'] if r['sample_id']==sample['sample_id'])
            available={op['id']:op for op in sample['operations']}
            available.update({'oracle_keep:'+b['id']:dict(parents=[b['id']],texts=[b['text']]) for b in sample['blocks']})
            chosen=[available[i] for i in oracle['selected_operation_ids']]
            if sorted(p for g in chosen for p in g['parents'])!=parent_ids:raise RuntimeError('Oracle input exclusivity violated')
            if sum((observed_tokens(chosen)&expected).values())!=oracle['correct']:raise RuntimeError('Oracle feasible count mismatch')
            checks['feasible_pool_oracle']+=1
        for arm,counts in aggregates.items():
            saved=result['cohorts'][phase]['arms'][arm]
            if any(saved[k]!=value for k,value in counts.items()):raise RuntimeError('Aggregate mismatch: '+arm)
            tp,n,g=counts['correct'],counts['predictions'],counts['gt_words']
            rates=dict(precision=tp/n if n else 0.,recall=tp/g if g else 0.,f1=2*tp/(n+g) if n+g else 0.)
            if any(abs(saved[k]-value)>1e-12 for k,value in rates.items()):raise RuntimeError('Aggregate rate mismatch')
        summaries[phase]={name:dict(value) for name,value in aggregates.items()}
    for name in ['completion_neural.json','denoiser_memory.json']:
        data=read(root/name);m=data.get('memory',data)
        if m['oom'] or m['peak_allocated_bytes']<=0 or m['peak_reserved_bytes']<m['peak_allocated_bytes']:
            raise RuntimeError('Invalid GPU qualification record')
    if (root/'failure.json').exists():raise RuntimeError('Full run contains an unresolved failure receipt')
    return dict(status='PASS',claim_label='approximate reproduction; independent saved-output audit',checks=dict(checks),
                result_sha256=file_hash(root/'results.json'),protocol_sha256=file_hash(root/'protocol.json'),
                summaries=summaries,training_updates=sum(r['updates'] for r in history['history']),
                limitations='Audit verifies source identities, counts, constraints and budgets, not independent scientific generalization.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=audit(args.run)
    with args.output.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ['status','checks','training_updates']}))


if __name__=='__main__':main()
