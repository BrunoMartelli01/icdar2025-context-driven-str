#!/usr/bin/env python3
"""Training-free PARSeq N-best contextual reranking diagnostic.

PRACTICAL_APPROXIMATION / TRAIN-ONLY DIAGNOSTIC.

Pipeline frozen up to PARSeq V1:
  frozen V1 crop -> frozen V1 transcription + PARSeq AR top-10 candidates
  -> Qwen2.5-1.5B likelihood scoring with:
       (a) surrounding OCR context only
       (b) surrounding OCR + fallible Florence-2-base-ft caption
  -> conservative V1-preserving gate

Qwen NEVER generates candidate text. Candidate strings are fixed before Qwen
inference and come only from V1 or the PARSeq AR beam.

Stages:
  prepare  : strip GT away from observable inputs; create evaluation labels separately
  infer    : GPU likelihood scoring only; no GT file is read
  evaluate : CPU TRAIN diagnostic; reports recovered/damaged and a small frozen grid
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DEFAULT_CANDIDATES = Path("artifacts/diagnostics/parseq-beam-candidate-coverage-train-v1/results.json")
DEFAULT_CONTEXT = Path("artifacts/diagnostics/geometry-context-reconstruction-train-v1/inputs.json")
DEFAULT_OUT = Path("artifacts/diagnostics/parseq-context-reranker-train-v1")
DEFAULT_HF_CACHE = Path("/extra1/bmartelli/context-str-artifacts/huggingface")
CONDITIONS = ("local", "local_florence")
TOPK = 10

# Small diagnostic grid. No model fitting occurs.
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)
TAUS = (0.0, 0.10, 0.25, 0.50, 1.00)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def context_samples(path: Path):
    obj = read_json(path)
    samples = obj.get("samples")
    if not isinstance(samples, list):
        raise RuntimeError(f"{path}: expected top-level samples list")
    out = {}
    for s in samples:
        sid = str(s["sample_id"])
        blocks = s.get("blocks")
        captions = s.get("captions")
        if not isinstance(blocks, list) or not isinstance(captions, dict):
            raise RuntimeError(f"context sample {sid}: expected blocks + captions")
        out[sid] = s
    return out


def make_observation(sample, target_cid):
    blocks = sample["blocks"]
    target_index = None
    observations = []
    for i, b in enumerate(blocks):
        cid = b["candidate_id"]
        if cid == target_cid:
            if target_index is not None:
                raise RuntimeError(f"duplicate target candidate id {target_cid}")
            target_index = i
            observations.append("<TARGET>")
        else:
            observations.append(str(b["text"]))
    if target_index is None:
        raise RuntimeError(f"target candidate not found in context blocks: {target_cid}")
    return observations, target_index


def dedupe_candidates(v1_text, ar_candidates):
    # V1 is always candidate 0 / KEEP and gets the strongest visual prior.
    cands = [{"text": v1_text, "source": "V1", "ar_rank": None}]
    seen = {v1_text.casefold()}
    for rank, row in enumerate(ar_candidates[:TOPK], 1):
        text = str(row["text"])
        # EOS-at-first-step can yield an empty AR hypothesis. It is not a
        # usable OCR transcription and can never match a non-empty ICDAR GT.
        if not text.strip():
            continue
        if text.casefold() in seen:
            continue
        seen.add(text.casefold())
        cands.append({
            "text": text,
            "source": "PARSEQ_AR_BEAM",
            "ar_rank": rank,
            "ar_logprob": float(row["logprob"]),
        })
    return cands


def prepare(args):
    cov = read_json(args.candidate_results)
    details = cov.get("details")
    if not isinstance(details, list) or len(details) != 745:
        raise RuntimeError("Expected candidate-coverage details for 745 eligible GT rows")
    if sum(bool(r["v1_exact"]) for r in details) != 502:
        raise RuntimeError("Expected frozen spatial V1 control 502/745")

    contexts = context_samples(args.context_inputs)
    grouped = defaultdict(list)
    for r in details:
        grouped[(str(r["sample_id"]), r["candidate_id"])].append(r)

    observable = []
    labels = []
    for (sid, cid), rows in grouped.items():
        ref = rows[0]
        v1 = ref["v1_text"]
        ar = ref["ar_candidates"]
        for r in rows[1:]:
            if r["v1_text"] != v1 or r["ar_candidates"] != ar:
                raise RuntimeError(f"inconsistent candidate evidence for crop {sid}/{cid}")
        sample = contexts[sid]
        observations, target_index = make_observation(sample, cid)
        caption = sample["captions"].get("florence2-base-ft")
        if not isinstance(caption, str) or not caption.strip():
            raise RuntimeError(f"missing Florence caption for sample {sid}")

        candidates = dedupe_candidates(v1, ar)
        observable.append({
            "sample_id": sid,
            "candidate_id": cid,
            "target_index": target_index,
            "observations_in_frozen_reading_order": observations,
            "florence_caption_fallible": caption,
            "v1_text": v1,
            "candidates": candidates,
        })
        for r in rows:
            labels.append({
                "sample_id": sid,
                "candidate_id": cid,
                "gt_index": int(r["gt_index"]),
                "gt_text": r["gt_text"],
                "v1_text": r["v1_text"],
                "v1_exact": bool(r["v1_exact"]),
            })

    if len(labels) != 745 or sum(x["v1_exact"] for x in labels) != 502:
        raise RuntimeError("label population/control mismatch")

    observable.sort(key=lambda x: (int(x["sample_id"]), x["candidate_id"]))
    labels.sort(key=lambda x: (int(x["sample_id"]), x["gt_index"]))

    protocol = {
        "classification": "PRACTICAL_APPROXIMATION_TRAIN_ONLY",
        "method": "PARSeq_AR_TOP10_CONTEXTUAL_LIKELIHOOD_RERANKING",
        "training": False,
        "test_loaded": False,
        "gt_in_inference_inputs": False,
        "candidate_source": "frozen V1 KEEP union PARSeq AR top-10",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16"},
        "conditions": list(CONDITIONS),
        "population": {
            "eligible_gt_rows": len(labels),
            "unique_v1_crops": len(observable),
            "v1_exact": 502,
        },
    }
    write_json(args.output_dir / "protocol.json", protocol)
    write_json(args.output_dir / "inputs.json", {"samples": observable})
    write_json(args.output_dir / "evaluation-labels.json", {"rows": labels})
    print(f"[prepare] {len(observable)} unique crops, 745 GT rows; V1=502/745")
    print("[prepare] observable inference inputs contain NO gt_text")


def build_user_prompt(sample, condition):
    obs = " | ".join(sample["observations_in_frozen_reading_order"])
    base = (
        "Score the visual OCR candidate that will follow using surrounding scene-text OCR only as weak linguistic context. "
        "The candidate itself was already proposed by the visual recognizer. "
        "Do not reconstruct, correct, or invent any different text. "
        "Higher probability should mean that this exact candidate fits the observed text context.\n"
        f"OCR observations in frozen reading order: {obs}\n"
    )
    if condition == "local_florence":
        base += (
            "Fallible scene caption (weak evidence only; it may be wrong and must not supply new characters): "
            + sample["florence_caption_fallible"].strip()
            + "\n"
        )
    elif condition != "local":
        raise ValueError(condition)
    base += "Exact target candidate:"
    return base


def candidate_score_batch(model, tokenizer, sample, device):
    import torch

    all_rows = []
    sequences = []
    spans = []

    for condition in CONDITIONS:
        user = build_user_prompt(sample, condition)
        prefix = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a conservative scene-text candidate scorer."},
                {"role": "user", "content": user},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(prefix, "tolist"):
            prefix = prefix.tolist()
        prefix = list(prefix)

        for ci, cand in enumerate(sample["candidates"]):
            cand_ids = tokenizer.encode(cand["text"], add_special_tokens=False)
            if not cand_ids:
                raise RuntimeError(f"empty candidate tokenization: {cand['text']!r}")
            seq = prefix + cand_ids
            sequences.append(seq)
            spans.append((len(prefix), len(cand_ids)))
            all_rows.append((condition, ci, cand["text"]))

    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    maxlen = max(map(len, sequences))
    input_ids = torch.full((len(sequences), maxlen), int(pad), dtype=torch.long, device=device)
    attn = torch.zeros((len(sequences), maxlen), dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attn[i, :len(seq)] = 1

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits.float()
        logp = torch.log_softmax(logits, dim=-1)

    out = {c: [] for c in CONDITIONS}
    for i, (condition, ci, text) in enumerate(all_rows):
        start, n = spans[i]
        # logits[t-1] predicts input_ids[t]
        pos = torch.arange(start, start + n, device=device)
        token_ids = input_ids[i, start:start+n]
        vals = logp[i, pos - 1, token_ids]
        out[condition].append({
            "candidate_index": ci,
            "text": text,
            "qwen_mean_logprob": float(vals.mean().item()),
            "qwen_sum_logprob": float(vals.sum().item()),
            "qwen_token_count": int(n),
        })
    return out


def snapshot_path(hf_cache: Path):
    p = hf_cache / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / MODEL_REVISION
    if not p.is_dir():
        raise FileNotFoundError(f"Qwen snapshot missing: {p}")
    return p


def infer(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    protocol = read_json(args.output_dir / "protocol.json")
    samples = read_json(args.output_dir / "inputs.json")["samples"]
    if protocol["gt_in_inference_inputs"]:
        raise RuntimeError("GT leakage flag is not false")

    snap = snapshot_path(args.hf_cache)
    tokenizer = AutoTokenizer.from_pretrained(snap, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        snap,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
    ).eval().to(args.device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    cache_dir = args.output_dir / "scores"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for n, sample in enumerate(samples, 1):
        out = cache_dir / f"{int(sample['sample_id']):04d}-{sample['candidate_id'].split(':')[-1]}.json"
        if out.exists():
            old = read_json(out)
            if old.get("candidate_id") == sample["candidate_id"] and old.get("v1_text") == sample["v1_text"]:
                # Reuse only if the candidate list is identical too. This makes
                # reruns safe after candidate filtering changes.
                if old.get("candidates") == sample["candidates"]:
                    print(f"[{n}/{len(samples)}] reuse {sample['sample_id']} {sample['candidate_id']}", flush=True)
                    continue

        if not str(sample["v1_text"]).strip():
            # Conservative fail-closed behavior: if V1 itself is empty, do not
            # let context invent a replacement. Keep V1 unchanged.
            keep_scores = {
                condition: [{
                    "candidate_index": 0,
                    "text": sample["v1_text"],
                    "qwen_mean_logprob": 0.0,
                    "qwen_sum_logprob": 0.0,
                    "qwen_token_count": 0,
                }]
                for condition in CONDITIONS
            }
            write_json(out, {
                "sample_id": sample["sample_id"],
                "candidate_id": sample["candidate_id"],
                "v1_text": sample["v1_text"],
                "candidates": [sample["candidates"][0]],
                "scores": keep_scores,
                "status": "force_keep_empty_v1",
            })
            print(f"[{n}/{len(samples)}] force-keep empty V1 {sample['sample_id']} {sample['candidate_id']}", flush=True)
            continue

        scores = candidate_score_batch(model, tokenizer, sample, torch.device(args.device))
        write_json(out, {
            "sample_id": sample["sample_id"],
            "candidate_id": sample["candidate_id"],
            "v1_text": sample["v1_text"],
            "candidates": sample["candidates"],
            "scores": scores,
        })
        print(f"[{n}/{len(samples)}] scored {sample['sample_id']} {sample['candidate_id']} candidates={len(sample['candidates'])}", flush=True)


def load_scores(args):
    samples = read_json(args.output_dir / "inputs.json")["samples"]
    out = {}
    for sample in samples:
        p = args.output_dir / "scores" / f"{int(sample['sample_id']):04d}-{sample['candidate_id'].split(':')[-1]}.json"
        if not p.is_file():
            raise RuntimeError(f"missing score shard {p}")
        row = read_json(p)
        out[(str(row["sample_id"]), row["candidate_id"])] = row
    return out


def visual_penalty(candidate):
    # KEEP/V1 is the default and has penalty 0.
    # Any alternative must pay a monotonically increasing penalty based on
    # its PARSeq AR beam rank. This avoids mixing incomparable raw score scales.
    if candidate["source"] == "V1":
        return 0.0
    rank = int(candidate["ar_rank"])
    return math.log1p(rank)


def choose(row, condition, lam, tau):
    cands = row["candidates"]
    scores = row["scores"][condition]
    if len(cands) != len(scores):
        raise RuntimeError("candidate/score cardinality mismatch")
    base_score = float(scores[0]["qwen_mean_logprob"])

    best_idx = 0
    best_gain = 0.0
    for i in range(1, len(cands)):
        qmargin = float(scores[i]["qwen_mean_logprob"]) - base_score
        gain = qmargin - lam * visual_penalty(cands[i])
        if gain > best_gain:
            best_gain = gain
            best_idx = i
    if best_idx != 0 and best_gain > tau:
        return best_idx, best_gain
    return 0, best_gain


def metrics(labels, scores, condition, lam, tau):
    chosen = {}
    gains = {}
    for key, row in scores.items():
        idx, gain = choose(row, condition, lam, tau)
        chosen[key] = row["candidates"][idx]["text"]
        gains[key] = gain

    exact = recovered = damaged = 0
    changed_crops = 0
    for key, row in scores.items():
        if chosen[key].casefold() != row["v1_text"].casefold():
            changed_crops += 1

    for lab in labels:
        key = (str(lab["sample_id"]), lab["candidate_id"])
        pred = chosen[key]
        ok = pred.casefold() == lab["gt_text"].casefold()
        exact += ok
        if ok and not lab["v1_exact"]:
            recovered += 1
        if lab["v1_exact"] and not ok:
            damaged += 1

    return {
        "condition": condition,
        "lambda": lam,
        "tau": tau,
        "exact": exact,
        "gain": exact - 502,
        "recovered": recovered,
        "damaged": damaged,
        "changed_crops": changed_crops,
    }


def per_crop_oracle(labels, scores):
    by = defaultdict(list)
    for lab in labels:
        by[(str(lab["sample_id"]), lab["candidate_id"])].append(lab)
    exact = 0
    for key, labs in by.items():
        texts = [c["text"] for c in scores[key]["candidates"]]
        best = 0
        for text in texts:
            best = max(best, sum(text.casefold() == x["gt_text"].casefold() for x in labs))
        exact += best
    return exact


def evaluate(args):
    labels = read_json(args.output_dir / "evaluation-labels.json")["rows"]
    if len(labels) != 745 or sum(x["v1_exact"] for x in labels) != 502:
        raise RuntimeError("frozen evaluation control mismatch")
    scores = load_scores(args)

    oracle = per_crop_oracle(labels, scores)
    grid = []
    for cond in CONDITIONS:
        for lam in LAMBDAS:
            for tau in TAUS:
                grid.append(metrics(labels, scores, cond, lam, tau))

    grid.sort(key=lambda x: (x["gain"], -x["damaged"], -x["recovered"]), reverse=True)
    best_by_cond = {}
    for cond in CONDITIONS:
        rows = [x for x in grid if x["condition"] == cond]
        best_by_cond[cond] = rows[0]

    # Predeclared reference settings for interpretation, not selected by GT.
    references = {
        "qwen_only_local": metrics(labels, scores, "local", 0.0, 0.0),
        "qwen_only_local_florence": metrics(labels, scores, "local_florence", 0.0, 0.0),
        "conservative_local": metrics(labels, scores, "local", 1.0, 0.25),
        "conservative_local_florence": metrics(labels, scores, "local_florence", 1.0, 0.25),
    }

    report = {
        "classification": "PRACTICAL_APPROXIMATION_TRAIN_ONLY_DIAGNOSTIC",
        "baseline_exact": 502,
        "eligible_gt_rows": 745,
        "candidate_oracle_per_crop_exact": oracle,
        "candidate_oracle_gain": oracle - 502,
        "references": references,
        "best_train_diagnostic_by_condition": best_by_cond,
        "grid": grid,
        "warning": "Best grid entries are TRAIN-selected diagnostics, not generalization claims. Qwen inference itself is training-free and GT-free.",
    }
    write_json(args.output_dir / "results.json", report)

    print("\n=== PARSEQ CONTEXT RERANKER ===")
    print(f"V1 baseline             502/745")
    print(f"candidate per-crop oracle {oracle}/745  gain={oracle-502:+d}")
    print("\nPREDECLARED SETTINGS")
    print("name                              exact  gain  rec  dam  changed")
    for name, r in references.items():
        print(f"{name:34s} {r['exact']:4d} {r['gain']:+5d} {r['recovered']:4d} {r['damaged']:4d} {r['changed_crops']:7d}")
    print("\nBEST TRAIN DIAGNOSTIC GRID")
    print("condition          lambda tau    exact  gain  rec  dam  changed")
    for cond in CONDITIONS:
        r = best_by_cond[cond]
        print(f"{cond:18s} {r['lambda']:5.2f} {r['tau']:4.2f}  {r['exact']:4d} {r['gain']:+5d} {r['recovered']:4d} {r['damaged']:4d} {r['changed_crops']:7d}")
    print("JSON", args.output_dir / "results.json")


def inspect(args):
    cov = read_json(args.candidate_results)
    contexts = context_samples(args.context_inputs)
    print("candidate rows", len(cov.get("details", [])))
    print("context samples", len(contexts))
    print("candidate control", cov.get("controls"))
    print("qwen snapshot", snapshot_path(args.hf_cache))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("inspect", "prepare", "infer", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--candidate-results", type=Path, default=DEFAULT_CANDIDATES)
        s.add_argument("--context-inputs", type=Path, default=DEFAULT_CONTEXT)
        s.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
        s.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE)
        s.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"inspect": inspect, "prepare": prepare, "infer": infer, "evaluate": evaluate}[a.stage](a)


if __name__ == "__main__":
    main()
