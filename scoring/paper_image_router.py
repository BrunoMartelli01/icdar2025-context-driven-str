"""APPROXIMATE image-level interpretation of paper section 3.5/Algorithm 2.

All crop words form T3. The paper's semantic candidate selection is recorded
separately from the localized-word gate used to compare fairly with CLIP.
No correctness labels, captions of other samples or confidence filters enter T3.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from scoring.lexical import lexical_similarity
from recognition.clip_router_prompt_ablation import transcript


@dataclass(frozen=True)
class PaperRouterConfig:
    alpha: float
    beta: float
    tau: float
    primary_gate: str
    lexical_mode: str

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in (self.alpha,self.beta,self.tau))
                or not 0<=self.alpha<=1 or not 0<=self.beta<=1
                or not math.isclose(self.alpha+self.beta,1) or not 0<=self.tau<=1
                or self.primary_gate not in ('paper_selected','joined_t3')
                or self.lexical_mode!='raw'):
            raise ValueError('Explicit approximate paper router configuration required')


def assemble_t3(rows):
    ids=set()
    for r in rows:
        if (not isinstance(r['raw_text'],str) or r['instance_id'] in ids
                or len(r['bbox'])!=4 or not all(math.isfinite(v) for v in r['bbox'])
                or r['bbox'][2]<=r['bbox'][0] or r['bbox'][3]<=r['bbox'][1]):
            raise ValueError('Invalid or duplicate crop prediction')
        ids.add(r['instance_id'])
    ordered=sorted(rows,key=lambda r:(r['bbox'][1],r['bbox'][0],r['instance_id']))
    return {'text':transcript(rows),'words':[
        {'instance_id':r['instance_id'],'bbox':r['bbox'],'raw_text':r['raw_text']} for r in ordered],
        'all_predictions_included':True,'deduplication':False,
        'assembly':'bbox top then left then instance ID; space join; same as CLIP',
        'eligible':bool(rows) and all(r['raw_text'].strip() for r in rows)}


def paper_decision(t1,t3,t2,s1,s3,config):
    if (not all(isinstance(t,str) for t in (t1,t3,t2))
            or not all(math.isfinite(s) and -1<=s<=1 for s in (s1,s3))):
        raise ValueError('Expected raw text and finite semantic scores')
    l1=lexical_similarity(t1,t2,config.lexical_mode).score
    l3=lexical_similarity(t3,t2,config.lexical_mode).score
    # Algorithm 2 line8 uses strict >, hence T3 wins exact semantic ties.
    winner='T1' if s1>s3 else 'T3'
    c1=config.alpha*s1+config.beta*l1;c3=config.alpha*s3+config.beta*l3
    confidence=c1 if winner=='T1' else c3
    selected=t1 if winner=='T1' else t3
    needs_fallback=not t2.strip() or not selected.strip() or confidence<config.tau
    return {'T1':t1,'T2':t2,'T3':t3,'S1':s1,'S3':s3,'L1':l1,'L3':l3,
        'C1':c1,'C3':c3,'C':confidence,'semantic_winner':winner,
        'selected_output':selected,'needs_fallback':needs_fallback,
        'alpha':config.alpha,'beta':config.beta,'tau':config.tau,
        'lexical_mode':config.lexical_mode,'lexical_function':'rapidfuzz.fuzz.ratio/100',
        'interpretation':'prose: confidence of semantic winner; Algorithm2 strict tie',
        'ground_truth_in_routing':False}


def localized_word_scores(decision,t3_assembly):
    """Threshold the selected candidate; T1 is one prediction, T3 is all crops."""
    eligible=t3_assembly['eligible'] and bool(decision['T2'].strip())
    selected_eligible=bool(decision['T2'].strip()) and (
        bool(decision['T1'].strip()) if decision['semantic_winner']=='T1' else t3_assembly['eligible'])
    return {'joined_t3':decision['C3'] if eligible else None,
        'paper_selected':decision['C'] if selected_eligible else None}


def caption_quality(text,min_words,max_words):
    words=text.split();triples=[tuple(words[i:i+3]) for i in range(max(0,len(words)-2))]
    return {'words':len(words),'characters':len(text),
        'sentences':sum(text.count(c) for c in '.!?'),
        'unique_trigram_fraction':len(set(triples))/len(triples) if triples else 1.,
        'within_target_words':min_words<=len(words)<=max_words,
        'length_target_is_assumption':True,'visual_correctness_automatically_verified':False}
