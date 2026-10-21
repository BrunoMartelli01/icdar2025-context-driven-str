import math
from types import SimpleNamespace

import pytest
import torch

from recognition.bert_visual_correction import CandidatePolicy,append_beam,ar_beam,ar_scores,candidate_pool,eligible,select


@pytest.fixture
def policy():
    return CandidatePolicy(2,25,6,1,3,0.5,32,96,10)


class ToyTokenizer:
    eos_id,bos_id,pad_id=0,3,4
    _stoi={"a":1,"b":2}

    def encode(self,texts,device=None):
        rows=[[3,*[self._stoi[c] for c in t],0] for t in texts]
        n=max(map(len,rows))
        return torch.tensor([r+[4]*(n-len(r)) for r in rows],device=device)

    def _ids2tok(self,ids):
        return "".join({1:"a",2:"b"}[i] for i in ids)


class ToyAR:
    max_label_length=4
    pos_queries=torch.arange(5).reshape(1,5,1)
    head=torch.nn.Identity()

    def decode(self,context,memory,tgt_mask=None,tgt_padding_mask=None,tgt_query=None,tgt_query_mask=None):
        # First-order autoregressive engineering fixture, independent of GT/images.
        positions=(range(context.shape[1]) if tgt_query is None else
                   [int(x) for x in tgt_query[0,:,0]])
        assert memory.shape[0]==context.shape[0]
        expected=torch.triu(torch.ones(context.shape[1],context.shape[1],dtype=torch.bool),1)
        assert torch.equal(tgt_mask.cpu(),expected)
        queries=list(positions)
        assert torch.equal(tgt_query_mask.cpu(),expected[queries])
        probabilities={3:[.05,.7,.25],1:[.6,.35,.05],2:[.7,.2,.1],0:[.8,.1,.1],4:[.8,.1,.1]}
        return torch.tensor([[probabilities[int(row[p])] for p in queries] for row in context]).log()


def test_candidate_external_lexicon_and_caption_no_gt(policy):
    result=candidate_pool("BRIDG",{},"A BRIDGE beside water",["bring","bridge","world"],"BRIDGE",policy)
    assert result[0]["text"]=="BRIDG"
    bridge=next(c for c in result if c["text"]=="BRIDGE")
    assert bridge["sources"]==["previous_insertion_anchor","Florence_caption_token","external_NeuSpell_lexicon"]
    assert "WATER" not in [c["text"] for c in result]


@pytest.mark.parametrize("text",["iPhone","12","AB-CD","two words","a", "a"*26])
def test_protected_shapes_keep_exact(text,policy):
    assert not eligible(text,policy)
    assert candidate_pool(text,{},"candidate",["candidate"],text,policy)==[
        {"text":text,"sources":["original_OCR_KEEP"],"edit_distance":0}]


def test_real_words_can_receive_visual_alternatives(policy):
    result=candidate_pool("sand",{"word":[{"text":"sand"},{"text":"sane"}]},"",["sand","send"],"sand",policy)
    assert [c["text"] for c in result]==["sand","sane","send"]
    beam=append_beam(result,[{"text":"surf"}],"sand",policy)
    assert beam[-1]["text"]=="surf" and beam[-1]["edit_distance"]>1


def test_immutable_insertion_anchor_survives_stricter_new_short_word_bound(policy):
    result=candidate_pool("SOCIE",{},"",[],"SOCIETY",policy)
    assert [c["text"] for c in result]==["SOCIE","SOCIETY"]


def test_teacher_forcing_matches_probability_and_eos():
    scores=ar_scores(ToyAR(),ToyTokenizer(),torch.zeros(1,1,1),["ab","a",""],2)
    assert scores[0]["token_count"]==3
    assert scores[0]["token_ids"]==[1,2,0]
    assert scores[0]["sum_log_probability"]==pytest.approx(math.log(.7*.05*.7),abs=1e-6)
    assert scores[1]["sum_log_probability"]==pytest.approx(math.log(.7*.6),abs=1e-6)
    assert scores[2]["sum_log_probability"]==pytest.approx(math.log(.05),abs=1e-6)


def test_complete_beam_agrees_with_independent_teacher_forcing():
    beam=ar_beam(ToyAR(),ToyTokenizer(),torch.zeros(1,1,1),16,2)
    assert len(beam)==7
    assert all(b["terminated"] and b["token_ids"][-1]==0 for b in beam)
    scores=ar_scores(ToyAR(),ToyTokenizer(),torch.zeros(1,1,1),[b["text"] for b in beam],3)
    for b,s in zip(beam,scores):
        assert b["sum_log_probability"]==pytest.approx(s["sum_log_probability"],abs=1e-6)
    assert beam[0]["text"]=="a"


def test_scores_independent_of_batch_padding_and_future_characters():
    model,tok,mem=ToyAR(),ToyTokenizer(),torch.zeros(1,1,1)
    batched=ar_scores(model,tok,mem,["abbb","a"],2)
    single=ar_scores(model,tok,mem,["a"],1)[0]
    assert batched[1]["token_log_probabilities"]==single["token_log_probabilities"]
    assert batched[0]["token_log_probabilities"][0]==single["token_log_probabilities"][0]


def test_unknown_and_overlength_candidates_fail():
    for t in ["c","aaaaa"]:
        with pytest.raises(ValueError,match="unsupported"):
            ar_scores(ToyAR(),ToyTokenizer(),torch.zeros(1,1,1),[t],1)


def test_reference_tie_and_visual_margin_protection():
    cs=[{"text":"original"},{"text":"candidate"}]
    vs=[{"sum_log_probability":-1.0},{"sum_log_probability":-2.1}]
    bs=[{"mean_log_probability":-5.0},{"mean_log_probability":-1.0}]
    assert select(cs,vs,bs,anchor_index=0,visual_margin=1,beta=.5,minimum_gain=.1)["selected_index"]==0
    vs[1]["sum_log_probability"]=-2.0
    assert select(cs,vs,bs,anchor_index=0,visual_margin=1,beta=.5,minimum_gain=.1)["selected_index"]==1
    vs[1]["sum_log_probability"]=-1.0
    assert select(cs,vs,None,anchor_index=1,visual_margin=0,beta=0,minimum_gain=0)["selected_index"]==1


def test_unknown_reference_abstains_and_nonfinite_fails():
    cs=[{"text":"keep"},{"text":"other"}]
    vs=[{"sum_log_probability":-1.0},{"sum_log_probability":-.5}]
    bs=[{"mean_log_probability":None},{"mean_log_probability":-.2}]
    assert select(cs,vs,bs,anchor_index=0,visual_margin=1,beta=.5,minimum_gain=.1)["reason"]=="context_reference_abstain"
    bs[0]["mean_log_probability"]=float("nan")
    with pytest.raises(ValueError,match="Nonfinite"):
        select(cs,vs,bs,anchor_index=0,visual_margin=1,beta=.5,minimum_gain=.1)


def test_invalid_candidate_policy():
    with pytest.raises(ValueError):
        CandidatePolicy(2,25,6,1,3,0.5,32,96,0)
