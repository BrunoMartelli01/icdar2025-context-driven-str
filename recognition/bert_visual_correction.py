"""Approximate visual/contextual correction. No ground truth enters this module."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from recognition.bert_method_selection import restore_case


@dataclass(frozen=True)
class CandidatePolicy:
    minimum_length: int
    maximum_length: int
    short_length: int
    short_edits: int
    long_edits: int
    maximum_relative_edit: float
    lexicon_candidates: int
    maximum_candidates: int
    beam_width: int

    def __post_init__(self):
        for name in ("minimum_length", "maximum_length", "short_length", "short_edits",
                     "long_edits", "lexicon_candidates", "maximum_candidates", "beam_width"):
            if type(getattr(self,name)) is not int or getattr(self,name) < 1:
                raise ValueError("Invalid candidate integer bound")
        if (self.maximum_length < self.minimum_length or self.maximum_candidates < 2
                or not math.isfinite(self.maximum_relative_edit)
                or not 0 < self.maximum_relative_edit <= 1):
            raise ValueError("Invalid candidate length/edit bound")


def eligible(text, policy):
    return (policy.minimum_length <= len(text) <= policy.maximum_length
            and re.fullmatch(r"[A-Za-z]+",text) is not None
            and (text.islower() or text.isupper() or text.istitle()))


def candidate_pool(original, heads, caption, vocabulary, insertion_anchor, policy):
    """Keep exact OCR first; source-independent, bounded, deterministic alternatives."""
    output=[{"text":original,"sources":["original_OCR_KEEP"],"edit_distance":0}]
    seen={original.casefold():0}
    if not eligible(original,policy):
        return output
    bound=policy.short_edits if len(original) < policy.short_length else policy.long_edits

    def add(text,source,bounded=True,preserve_case=False):
        if (not re.fullmatch(r"[A-Za-z]+",text)
                or not policy.minimum_length <= len(text) <= policy.maximum_length):
            return
        text=text if preserve_case else restore_case(text,original)
        distance=Levenshtein.distance(original.casefold(),text.casefold())
        if bounded and (distance > bound or distance/len(original) > policy.maximum_relative_edit):
            return
        key=text.casefold()
        if key in seen:
            sources=output[seen[key]]["sources"]
            if source not in sources:
                sources.append(source)
        elif len(output) < policy.maximum_candidates:
            seen[key]=len(output)
            output.append({"text":text,"sources":[source],"edit_distance":distance})

    # This is an immutable measured control, not a newly generated alternative.
    # Its previous two-insertion rule may exceed the new short-word edit bound.
    add(insertion_anchor,"previous_insertion_anchor",bounded=False,preserve_case=True)
    for source,labels in heads.items():
        for label in labels:
            add(label["text"],"NeuSpell/"+source)
    for word in re.findall(r"[A-Za-z]+",caption):
        add(word,"Florence_caption_token")
    # The supplied external lexicon preserves authenticated NeuSpell label order.
    for word,_,_ in process.extract(original.casefold(),vocabulary,
            scorer=Levenshtein.distance,score_cutoff=bound,limit=policy.lexicon_candidates):
        add(word,"external_NeuSpell_lexicon")
    return output


def append_beam(candidates,beam,original,policy):
    """Visual candidates may exceed lexical edit bounds, but still protect shape."""
    output=[{**c,"sources":list(c["sources"])} for c in candidates]
    if not eligible(original,policy):
        return output
    seen={c["text"].casefold():i for i,c in enumerate(output)}
    for row in beam:
        text=row["text"]
        if not (re.fullmatch(r"[A-Za-z]+",text) and
                policy.minimum_length <= len(text) <= policy.maximum_length):
            continue
        key=text.casefold()
        if key in seen:
            if "PARSeq_AR_beam" not in output[seen[key]]["sources"]:
                output[seen[key]]["sources"].append("PARSeq_AR_beam")
        elif len(output) < policy.maximum_candidates:
            seen[key]=len(output)
            output.append({"text":text,"sources":["PARSeq_AR_beam"],
                "edit_distance":Levenshtein.distance(original.casefold(),key)})
    return output


def ar_scores(model,tokenizer,memory,texts,batch_size):
    """Canonical causal teacher forcing, including EOS, never cloze refinement."""
    import torch
    if batch_size < 1 or not texts:
        raise ValueError("AR scoring requires texts and a positive batch size")
    output=[]
    for start in range(0,len(texts),batch_size):
        group=texts[start:start+batch_size]
        if any(len(t)>model.max_label_length or any(c not in tokenizer._stoi for c in t) for t in group):
            raise ValueError("Candidate unsupported by qualified PARSeq length/charset")
        tokens=tokenizer.encode(group,device=memory.device)
        context,targets=tokens[:,:-1],tokens[:,1:]
        length=context.shape[1]
        mask=torch.triu(torch.ones(length,length,dtype=torch.bool,device=memory.device),1)
        with torch.inference_mode():
            logits=model.head(model.decode(context,memory.expand(len(group),-1,-1),
                tgt_mask=mask,tgt_padding_mask=context==tokenizer.pad_id,tgt_query_mask=mask))
            safe_targets=targets.masked_fill(targets==tokenizer.pad_id,tokenizer.eos_id)
            logs=logits.float().log_softmax(-1).gather(-1,safe_targets.unsqueeze(-1)).squeeze(-1)
        for index,text in enumerate(group):
            values=logs[index,:len(text)+1].cpu().tolist()
            if len(values)!=len(text)+1 or any(not math.isfinite(v) for v in values):
                raise RuntimeError("Nonfinite/incomplete PARSeq candidate scores")
            output.append({"status":"scored","sum_log_probability":sum(values),
                "mean_log_probability":sum(values)/len(values),"token_log_probabilities":values,
                "token_count":len(values),"token_ids":tokens[index,1:len(text)+2].cpu().tolist(),
                "decoder_input_shape":list(context.shape),"memory_shape":list(memory.shape)})
    return output


def ar_beam(model,tokenizer,memory,width,max_chars):
    """Deterministic global beam; every returned sequence terminates with scored EOS."""
    import torch
    if width < 1 or not 1 <= max_chars <= model.max_label_length or memory.shape[0]!=1:
        raise ValueError("Invalid AR beam width/length/memory")
    active=[((),0.0,())]
    finished=[]
    for step in range(max_chars+1):
        if not active:
            break
        context=torch.tensor([[tokenizer.bos_id,*ids] for ids,_,_ in active],
                             device=memory.device,dtype=torch.long)
        mask=torch.triu(torch.ones(step+1,step+1,dtype=torch.bool,device=memory.device),1)
        query=model.pos_queries[:,step:step+1].expand(len(active),-1,-1)
        with torch.inference_mode():
            logs=model.head(model.decode(context,memory.expand(len(active),-1,-1),
                tgt_mask=mask,tgt_query=query,tgt_query_mask=mask[step:step+1])).float().log_softmax(-1)[:,0]
        values=logs.cpu().tolist()
        proposals=[(ids,score,path,True) for ids,score,path in finished]
        for (ids,score,path),probabilities in zip(active,values):
            if any(not math.isfinite(v) for v in probabilities):
                raise RuntimeError("Nonfinite PARSeq beam probability")
            tokens=([tokenizer.eos_id] if step==max_chars else
                sorted(range(len(probabilities)),key=lambda t:(-probabilities[t],t))[:width])
            for token in tokens:
                value=probabilities[token]
                end=token==tokenizer.eos_id
                proposals.append((ids if end else (*ids,token),score+value,(*path,value),end))
        proposals.sort(key=lambda p:(-p[1],p[0],not p[3]))
        proposals=proposals[:width]
        active=[(ids,score,path) for ids,score,path,end in proposals if not end]
        finished=[(ids,score,path) for ids,score,path,end in proposals if end]
    if active or not finished:
        raise RuntimeError("AR beam did not terminate with EOS")
    return [{"text":tokenizer._ids2tok(list(ids)),"token_ids":[*ids,tokenizer.eos_id],
        "sum_log_probability":score,"token_log_probabilities":list(path),"terminated":True}
        for ids,score,path in sorted(finished,key=lambda p:(-p[1],p[0]))]


def select(candidates,visual,context,*,anchor_index,visual_margin,beta,minimum_gain):
    """Compare real candidate/reference scores; ties and unavailable context KEEP."""
    if not 0 <= anchor_index < len(candidates) or len(visual)!=len(candidates):
        raise ValueError("Invalid visual selection cardinality/anchor")
    if (not all(math.isfinite(x) and x>=0 for x in (visual_margin,beta,minimum_gain))
            or beta and (context is None or len(context)!=len(candidates))):
        raise ValueError("Invalid contextual selector parameters/cardinality")
    ref=visual[anchor_index]["sum_log_probability"]
    if not math.isfinite(ref):
        raise ValueError("Nonfinite AR reference")
    if beta and context[anchor_index]["mean_log_probability"] is None:
        return {"selected_index":anchor_index,"reason":"context_reference_abstain","gain":0.0}
    cref=context[anchor_index]["mean_log_probability"] if beta else 0.0
    if not math.isfinite(cref):
        raise ValueError("Nonfinite BERT reference")
    best,greatest=anchor_index,minimum_gain
    for index,v in enumerate(visual):
        value=v["sum_log_probability"]
        if not math.isfinite(value):
            raise ValueError("Nonfinite AR candidate")
        dv=value-ref
        if dv < -visual_margin:
            continue
        dc=0.0
        if beta:
            c=context[index]["mean_log_probability"]
            if c is None:
                continue
            if not math.isfinite(c):
                raise ValueError("Nonfinite BERT candidate")
            dc=c-cref
        gain=dv+beta*dc
        if index!=anchor_index and gain>greatest:
            best,greatest=index,gain
    return {"selected_index":best,"reason":"visual_context_gain" if best!=anchor_index else "KEEP",
            "gain":greatest if best!=anchor_index else 0.0}
