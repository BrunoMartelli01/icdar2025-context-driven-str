"""APPROXIMATE literal-caption evidence and label-free contrastive text probes.

These functions never inspect annotations and do not train/load any model.
Caption text is an imperfect observation, never an exhaustive GT transcript.
All experimental limits and edit-match thresholds are explicit caller inputs.
"""
from __future__ import annotations

from collections import Counter
import math
import re
from typing import Callable, Mapping, Sequence
import unicodedata

import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment


QUOTE_PAIRS = {'"':'"', "'":"'", '“':'”', '‘':'’', '«':'»', '‹':'›', '「':'」', '『':'』', '`':'`'}
_GENERIC_EVIDENCE = {
    'text', 'words', 'word', 'letters', 'letter', 'written', 'printed', 'visible',
    'displayed', 'some text', 'a word', 'a message', 'the word', 'the words',
    'a sign', 'the sign', 'on it', 'something', 'a few words',
}
_ATTRIBUTION_PATTERNS = (
    ('word_on_surface', re.compile(
        r'\b(?:word|words|text|letter|letters)\s+(?P<evidence>.+?)\s+'
        r'(?P<boundary>on\s+(?:it|the\s+(?:sign|label|banner|poster|board|window|wall)))\b', re.I)),
    ('sign_says', re.compile(
        r'\b(?:sign|label|banner|poster|board|window)\s+(?:that\s+)?(?:says|reads)\s+'
        r'(?P<evidence>[^\n]+?)(?=\s+(?:with|next\s+to|in\s+front\s+of|above|below|and\s+a)\b|[.!?](?:\s|$)|[,;]|$)', re.I)),
    ('reads', re.compile(
        r'\breads\s+(?P<evidence>[^\n]+?)'
        r'(?=\s+(?:with|next\s+to|in\s+front\s+of|above|below|and\s+a)\b|[.!?](?:\s|$)|[,;]|$)', re.I)),
)


def _word_character(value: str) -> bool:
    return bool(value) and (value == '_' or unicodedata.category(value)[0] in 'LNM')


def _escaped(text: str, index: int) -> bool:
    before=index-1
    while before>=0 and text[before]=='\\':before-=1
    return (index-before-1)%2==1


def _quoted_spans(caption: str) -> list[dict]:
    spans=[];i=0
    while i<len(caption):
        opening=caption[i]
        if opening not in QUOTE_PAIRS or _escaped(caption,i):i+=1;continue
        # Contractions and possessives such as there's / dogs' are not open quotes.
        if opening in ('\'', '"', '`') and i and _word_character(caption[i-1]):i+=1;continue
        closing=QUOTE_PAIRS[opening];j=i+1
        while j<len(caption):
            if caption[j]!=closing or _escaped(caption,j):j+=1;continue
            if closing=="'" and j+1<len(caption) and _word_character(caption[j+1]):j+=1;continue
            text=caption[i+1:j]
            if text.strip():
                spans.append({'text':text,'start':i+1,'end':j,'delimiter_start':i,'delimiter_end':j+1,
                    'opening':opening,'closing':closing,'source':'quoted','ambiguous':False})
            i=j+1;break
        else:i+=1
    return spans


def _evidence_record(spans: Sequence[dict], *, rejected: Sequence[dict], source: str) -> dict:
    tokens=[]
    for index,span in enumerate(spans):
        for token in re.finditer(r'\S+',span['text']):
            raw=token.group(0)
            tokens.append({'text':raw,'casefold':raw.casefold(),'start':span['start']+token.start(),
                'end':span['start']+token.end(),'span_index':index,'source':span['source']})
    values=[t['casefold'] for t in tokens]
    return {'spans':list(spans),'tokens':tokens,'rejected_candidates':list(rejected),
        'evidence_present':bool(tokens),'evidence_source':source,
        'ambiguous_attribution_present':any(s['ambiguous'] for s in spans),
        'caption_repeats_evidence':any(n>1 for n in Counter(values).values()),
        'caption_may_hallucinate':True,'caption_is_exhaustive_transcript':False,
        'ground_truth_used':False,'tokenization':'caption-span whitespace; OCR instances are not split; casefold only; punctuation retained'}


def quoted_caption_evidence(caption: str) -> dict:
    if not isinstance(caption,str):raise ValueError('Caption must be a string')
    return _evidence_record(_quoted_spans(caption),rejected=[],source='quoted_only')


def explicit_caption_evidence(caption: str) -> dict:
    """Quoted text plus narrow English attribution patterns, with raw span provenance.

    Unquoted attribution is always marked ambiguous. Generic caption prose,
    including 'a sign with text on it', is not treated as a transcript.
    """
    if not isinstance(caption,str):raise ValueError('Caption must be a string')
    spans=_quoted_spans(caption);rejected=[]
    for pattern_name,pattern in _ATTRIBUTION_PATTERNS:
        for match in pattern.finditer(caption):
            start,end=match.span('evidence');raw=caption[start:end]
            leading=len(raw)-len(raw.lstrip());trailing=len(raw)-len(raw.rstrip())
            start+=leading;end-=trailing;text=caption[start:end]
            candidate={'text':text,'start':start,'end':end,'source':pattern_name,
                'trigger_start':match.start(),'trigger_end':match.end(),'ambiguous':True}
            reason=None
            if any(start<s['end'] and end>s['start'] for s in spans):reason='overlaps_preferred_evidence_span'
            elif not text or text.casefold() in _GENERIC_EVIDENCE:reason='generic_nonliteral_phrase'
            elif re.match(r'^(?:some|a\s+few)\s+(?:text|words|letters)\b',text,re.I):reason='generic_nonliteral_phrase'
            if reason:rejected.append({**candidate,'reason':reason})
            else:spans.append(candidate)
    spans.sort(key=lambda s:(s['start'],s['end'],s['source']))
    return _evidence_record(spans,rejected=rejected,source='quoted_or_explicit_attribution')


def caption_evidence_features(ocr_words: Sequence[str], evidence: Mapping, *, minimum_edit_similarity: float) -> dict:
    """Agreement with caption evidence, not recognition P/R/F1 against annotations."""
    if (not 0<minimum_edit_similarity<=1 or not math.isfinite(minimum_edit_similarity)
            or isinstance(ocr_words,(str,bytes)) or any(not isinstance(w,str) for w in ocr_words)):
        raise ValueError('Explicit OCR word instances and finite edit threshold in(0,1] required')
    if evidence.get('ground_truth_used') is not False or evidence.get('caption_is_exhaustive_transcript') is not False:
        raise ValueError('Expected imperfect, non-exhaustive caption evidence')
    words=[w.casefold() for w in ocr_words];tokens=[t['casefold'] for t in evidence['tokens']]
    result={'ocr_words':len(words),'caption_evidence_tokens':len(tokens),'evidence':dict(evidence),
        'evidence_present':bool(tokens),'ground_truth_used':False,'can_certify_completeness':False,
        'caption_may_hallucinate':True,'ocr_empty':not words,'minimum_edit_similarity':minimum_edit_similarity}
    if not tokens:
        return {**result,'exact_matches':None,'edit_matches':None,'exact_ocr_support':None,
            'exact_caption_recovery':None,'exact_agreement_f1':None,'edit_ocr_support':None,
            'edit_caption_recovery':None,'soft_ocr_support':None,'soft_caption_recovery':None,
            'unmatched_ocr_words':None,'unmatched_caption_evidence_tokens':None,'edit_pairs':[]}
    exact=sum((Counter(w for w in words if w)&Counter(tokens)).values())
    weights=np.full((len(words),len(tokens)+len(words)),-np.inf)
    similarities=np.zeros((len(words),len(tokens)),dtype=np.float64)
    for i,w in enumerate(words):
        weights[i,len(tokens)+i]=0
        for j,t in enumerate(tokens):
            similarity=float(Levenshtein.normalized_similarity(w,t)) if w and t else 0.
            similarities[i,j]=similarity
            if similarity>=minimum_edit_similarity:weights[i,j]=1+min(len(words),len(tokens))+similarity
    if words:
        ii,jj=linear_sum_assignment(weights,maximize=True)
        pairs=[{'ocr_index':int(i),'evidence_index':int(j),'similarity':float(similarities[i,j])}
            for i,j in zip(ii,jj) if j<len(tokens)]
    else:pairs=[]
    soft=sum(p['similarity'] for p in pairs);matched=len(pairs)
    return {**result,'exact_matches':exact,'edit_matches':matched,
        'exact_ocr_support':exact/len(words) if words else None,'exact_caption_recovery':exact/len(tokens),
        'exact_agreement_f1':2*exact/(len(words)+len(tokens)),
        'edit_ocr_support':matched/len(words) if words else None,'edit_caption_recovery':matched/len(tokens),
        'soft_ocr_support':soft/len(words) if words else None,'soft_caption_recovery':soft/len(tokens),
        'unmatched_ocr_words':len(words)-exact,'unmatched_caption_evidence_tokens':len(tokens)-exact,'edit_pairs':pairs}


def quoted_caption_features(caption: str, ocr_words: Sequence[str], *, minimum_edit_similarity: float) -> dict:
    return caption_evidence_features(ocr_words,quoted_caption_evidence(caption),minimum_edit_similarity=minimum_edit_similarity)


def _corrupt_word(text: str) -> str:
    positions=[i for i,c in enumerate(text) if not c.isspace()]
    if not positions:return text
    i=positions[len(positions)//2];replacement='x' if text[i].casefold()!='x' else 'z'
    return text[:i]+replacement+text[i+1:]


def contrastive_transcript_variants(words: Sequence[str], *, max_words: int, max_variants: int, max_characters: int) -> dict:
    """Bound probe count without truncating the original or silently omitting words.

    Over-limit originals remain present, with an explicit UNAVAILABLE status
    and no alternative probes. All limits are experiment configuration inputs.
    """
    if (isinstance(words,(str,bytes)) or any(not isinstance(w,str) for w in words)
            or min(max_words,max_variants,max_characters)<1):
        raise ValueError('Explicit string word instances and positive bounds required')
    words=list(words);original=' '.join(words)
    first={'name':'original','text':original,'control_type':'original','source_word_indices':list(range(len(words)))}
    exceeded=[]
    if len(words)>max_words:exceeded.append('word_count')
    if len(original)>max_characters:exceeded.append('character_count')
    if exceeded:
        return {'status':'PROBES_UNAVAILABLE_ORIGINAL_EXCEEDS_LIMIT','variants':[first],
            'original_preserved':True,'limits_exceeded':exceeded,'word_count':len(words),'character_count':len(original)}
    candidates=[first]
    if any(words):
        candidates.append({'name':'character_corruption','text':' '.join(_corrupt_word(w) for w in words),
            'control_type':'spelling_negative','source_word_indices':list(range(len(words)))})
    if len(words)>1:
        for parity in (0,1):
            indices=list(range(parity,len(words),2))
            candidates.append({'name':f'alternating_subset_{parity}','text':' '.join(words[i] for i in indices),
                'control_type':'deletion_subset','source_word_indices':indices})
    # Spread one-word deletions over the original sequence when probe budget is tight.
    remaining=max(0,max_variants-len(candidates))
    indices=sorted({round(i*(len(words)-1)/max(1,remaining-1)) for i in range(min(remaining,len(words)))}) if words else []
    for index in indices:
        keep=[i for i in range(len(words)) if i!=index]
        candidates.append({'name':f'delete_word_{index}','text':' '.join(words[i] for i in keep),
            'control_type':'deletion_one_word','source_word_indices':keep,'deleted_word_index':index})
    result=[];seen=set()
    for candidate in candidates:
        # Exact transcript duplicates are not independent negative controls.
        if candidate['text'] in seen:continue
        seen.add(candidate['text']);result.append(candidate)
        if len(result)==max_variants:break
    return {'status':'READY' if words else 'EMPTY_OCR','variants':result,'original_preserved':True,
        'limits_exceeded':[],'word_count':len(words),'character_count':len(original),
        'probe_count_capped':len(result)<len(candidates),'ground_truth_used':False}


def other_image_controls(transcripts: Mapping[str,str], *, controls_per_image: int) -> dict[str,list[dict]]:
    """Stable ID-cyclic negative transcripts from the current population, no GT."""
    if controls_per_image<1 or any(not isinstance(k,str) or not isinstance(v,str) for k,v in transcripts.items()):
        raise ValueError('Stable string IDs/transcripts and positive control count required')
    ids=sorted(transcripts);result={}
    for i,sid in enumerate(ids):
        rows=[];seen={transcripts[sid].casefold()}
        for offset in range(1,len(ids)):
            other=ids[(i+offset)%len(ids)];text=transcripts[other]
            if not text.strip() or text.casefold() in seen:continue
            rows.append({'sample_id':other,'text':text,'control_type':'other_image_transcript'})
            seen.add(text.casefold())
            if len(rows)==controls_per_image:break
        result[sid]=rows
    return result


def validate_token_budget(texts: Sequence[str], *, token_counter: Callable[[str],int], token_limit: int) -> list[int]:
    """Reject over-context prompts explicitly; do not truncate caption/transcript."""
    if token_limit<1 or isinstance(texts,(str,bytes)) or any(not isinstance(t,str) for t in texts):
        raise ValueError('Prompt strings and a positive explicit token limit required')
    counts=[token_counter(text) for text in texts]
    if any(type(n) is not int or n<0 for n in counts):raise ValueError('Token counter must return nonnegative integers')
    exceeded=[i for i,n in enumerate(counts) if n>token_limit]
    if exceeded:raise ValueError(f'Prompt token limit exceeded at indices {exceeded}; refusing truncation')
    return counts
