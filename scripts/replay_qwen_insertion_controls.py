"""Fixed insertion controls on historical Qwen outputs; no new model inference."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from recognition.text_reconstruction import insertion_candidate


def replay(source, conditions, filters):
    baseline=source['per_word']['V1']
    identities=[(r['sample_id'],r['gt_index']) for r in baseline]
    if len(identities)!=849 or len(set(identities))!=849:
        raise ValueError('Expected the frozen 849-word historical population')
    originals={r['candidate_id']:r['accepted'] for r in source['crop_decisions']['V1']}
    if len(originals)!=1203:raise ValueError('Historical crop population changed')
    output=[]
    for condition in conditions:
        rows=source['per_word'][condition]
        if [(r['sample_id'],r['gt_index']) for r in rows]!=identities:
            raise ValueError('Condition word identities differ')
        for gate in filters:
            # The decision is made once for every input crop, without word labels.
            decisions={}
            for row in source['crop_decisions'][condition]:
                cid=row['candidate_id'];old=originals[cid];new=row['accepted']
                decisions[cid]=new if insertion_candidate(old,new,maximum_insertions=gate['maximum_insertions'],minimum_length=gate['minimum_length']) else old
            if set(decisions)!=set(originals):raise ValueError('Condition crop identities differ')
            exact=recovered=damaged=0;outcomes=[];per_image={}
            for base,row in zip(baseline,rows):
                if row['before']!=base['before'] or row['gt_text']!=base['gt_text']:
                    raise ValueError('Historical baseline/labels differ')
                cid=row['candidate_id'];chosen=decisions[cid] if cid is not None else None
                before=isinstance(base['before'],str) and base['before'].casefold()==row['gt_text'].casefold()
                after=isinstance(chosen,str) and chosen.casefold()==row['gt_text'].casefold()
                exact+=after;recovered+=after and not before;damaged+=before and not after
                image=per_image.setdefault(row['sample_id'],dict(exact=0,recovered=0,damaged=0,gt_words=0))
                for key,value in [('exact',after),('recovered',after and not before),('damaged',before and not after),('gt_words',1)]:image[key]+=value
                if after!=before:outcomes.append(dict(sample_id=row['sample_id'],gt_index=row['gt_index'],gt_text=row['gt_text'],before=base['before'],after=chosen,recovered=after))
            if exact!=502+recovered-damaged:raise RuntimeError('Replay accounting failed')
            output.append(dict(condition=condition,filter=gate,exact=exact,recovered=recovered,damaged=damaged,
                               changed_crops=sum(v!=originals[k] for k,v in decisions.items()),per_image=per_image,evaluation_outcomes=outcomes))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    if repo/'configs/approximate' not in args.config.resolve().parents:raise ValueError('Isolated approximate config required')
    cfg=json.loads(args.config.read_text())
    if cfg['reproduction_mode']!='approximate':raise ValueError('Approximate mode required')
    raw=Path(cfg['source']).read_bytes()
    if sha256(raw).hexdigest()!=cfg['source_sha256']:raise ValueError('Historical report hash changed')
    results=replay(json.loads(raw),cfg['conditions'],cfg['filters'])
    report=dict(reproduction_mode='approximate',claim_label='approximate reproduction',result_class='post-hoc saved-output diagnostic',
                resolved_config=cfg,results=results,new_neural_inference=False,independent_validation=False,
                warning='Policies/conditions follow previously inspected outcomes. This reproduces a hypothesis, not independent validation; all849 care GT remain.')
    args.output.mkdir(exist_ok=False,parents=True)
    (args.output/'results.json').write_text(json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['condition','filter','exact','recovered','damaged']} for r in results]),flush=True)


if __name__=='__main__':main()
