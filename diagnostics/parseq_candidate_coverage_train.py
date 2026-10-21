#!/usr/bin/env python3
"""TRAIN-only PARSeq candidate-coverage diagnostic.

DIAGNOSTIC_ONLY:
- frozen PRACTICAL_REMAP_V1 crops;
- no crop/localization changes;
- no context model and no correction;
- GT is used ONLY after candidate generation to measure oracle coverage.

Candidate sources:
1) AR beam before PARSeq iterative refinement;
2) k-best lattice from PARSeq final (post-refinement) logits.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

PARSEQ_REVISION = "1902db043c029a7e03a3818c616c06600af574be"
PARSEQ_CKPT_NAME = "parseq-bb5792a6.pt"
DEFAULT_AUDIT = Path("artifacts/diagnostics/train-v1-ocr-failure-audit-v1/results.json")
DEFAULT_OUT = Path("artifacts/diagnostics/parseq-beam-candidate-coverage-train-v1")
KS = (1, 3, 5, 10, 20)


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def discover_compact() -> Path:
    direct = [
        Path("artifacts/diagnostics/train-practical-remap-v1-parseq-compact-v1"),
        Path("/extra1/bmartelli/context-str-runs/train-practical-remap-v1-parseq-compact-v1"),
    ]
    for p in direct:
        if (p / "manifest.json").is_file():
            return p.resolve()
    roots = [Path("/extra1/bmartelli/context-str-runs"), Path("artifacts")]
    hits = []
    for root in roots:
        if root.exists():
            hits += [p.parent for p in root.glob("**/train-practical-remap-v1-parseq-compact-v1/manifest.json")]
    uniq = sorted({p.resolve() for p in hits})
    if len(uniq) == 1:
        return uniq[0]
    raise RuntimeError("Cannot uniquely discover compact V1 root. Pass --compact-root. Hits=" + repr([str(x) for x in uniq[:20]]))


def discover_parseq_source() -> Path:
    name = "parseq-" + PARSEQ_REVISION
    roots = [
        Path("/extra1/bmartelli/context-str-artifacts/parseq"),
        Path("/extra1/bmartelli/context-str-artifacts"),
    ]
    for root in roots:
        direct = root / name
        if direct.is_dir():
            return direct.resolve()
    hits = []
    root = Path("/extra1/bmartelli/context-str-artifacts")
    if root.exists():
        hits = [p for p in root.glob("**/" + name) if p.is_dir()]
    uniq = sorted({p.resolve() for p in hits})
    if len(uniq) == 1:
        return uniq[0]
    raise RuntimeError("Cannot uniquely discover pinned PARSeq source. Pass --parseq-source. Hits=" + repr([str(x) for x in uniq[:20]]))


def discover_parseq_checkpoint() -> Path:
    roots = [
        Path("/extra1/bmartelli/context-str-artifacts/parseq"),
        Path("/extra1/bmartelli/context-str-artifacts"),
    ]
    for root in roots:
        p = root / PARSEQ_CKPT_NAME
        if p.is_file():
            return p.resolve()
    hits = []
    root = Path("/extra1/bmartelli/context-str-artifacts")
    if root.exists():
        hits = [p for p in root.glob("**/" + PARSEQ_CKPT_NAME) if p.is_file()]
    uniq = sorted({p.resolve() for p in hits})
    if len(uniq) == 1:
        return uniq[0]
    raise RuntimeError("Cannot uniquely discover PARSeq checkpoint. Pass --parseq-checkpoint. Hits=" + repr([str(x) for x in uniq[:20]]))


def load_population(compact: Path, audit_path: Path):
    audit = read_json(audit_path)
    rows = audit["per_word"]
    if len(rows) != 849:
        raise RuntimeError(f"Expected TRAIN849 GT rows, got {len(rows)}")

    bbox_to_cid = {}
    cached_ocr = {}
    crop_paths = {}
    for p in sorted((compact / "candidate-shards").glob("*.json")):
        c = read_json(p)
        sid = str(c["sample_id"])
        o = read_json(compact / "ocr-shards" / p.name)
        by_id = {x["candidate_id"]: x for x in o["results"]}
        for x in c["candidates"]:
            cid = x["candidate_id"]
            bbox_to_cid[(sid, tuple(x["bbox"]))] = cid
            result = by_id[cid]
            if "raw_text" in result:
                cached_text = result["raw_text"]
            elif "text" in result:
                cached_text = result["text"]
            else:
                raise RuntimeError(
                    f"PARSeq shard result has no raw_text/text field: candidate={cid} keys={sorted(result)}"
                )
            cached_ocr[(sid, cid)] = cached_text
            crop_paths[(sid, cid)] = compact / x["crop_path"]

    selected = []
    for r in rows:
        if not r.get("available") or r.get("crop_bbox") is None:
            continue
        sid = str(r["sample_id"])
        key = (sid, tuple(r["crop_bbox"]))
        if key not in bbox_to_cid:
            raise RuntimeError(f"Audit crop bbox not found in compact candidates: sample={sid} bbox={r['crop_bbox']}")
        cid = bbox_to_cid[key]
        cached = cached_ocr[(sid, cid)]
        if cached != r["parseq_text"]:
            raise RuntimeError(f"Frozen PARSeq control mismatch sample={sid} cid={cid}: compact={cached!r} audit={r['parseq_text']!r}")
        selected.append({
            "sample_id": sid,
            "gt_index": int(r["gt_index"]),
            "gt_text": r["gt_text"],
            "v1_text": r["parseq_text"],
            "v1_exact": bool(r["exact"]),
            "candidate_id": cid,
            "crop_path": str(crop_paths[(sid, cid)]),
        })
    return selected


def decode_tokens(model, tokens):
    if not tokens:
        return ""
    nclass = int(model.head.out_features)
    probs = torch.zeros((1, len(tokens), nclass), dtype=torch.float32)
    for i, t in enumerate(tokens):
        if t < 0 or t >= nclass:
            raise RuntimeError(f"Unexpected token id {t}, nclass={nclass}")
        probs[0, i, t] = 1.0
    labels, _ = model.tokenizer.decode(probs)
    return labels[0]


def unique_texts(model, seqs, limit):
    out, seen = [], set()
    for tokens, score in sorted(seqs, key=lambda z: z[1], reverse=True):
        text = decode_tokens(model, tokens)
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append({"text": text, "logprob": float(score)})
        if len(out) >= limit:
            break
    return out


@torch.inference_mode()
def ar_beam(model, image_tensor, beam_width=20):
    device = image_tensor.device
    memory = model.encode(image_tensor)
    num_steps = int(model.max_label_length) + 1
    pos_queries = model.pos_queries[:, :num_steps]
    mask = torch.triu(torch.full((num_steps, num_steps), True, dtype=torch.bool, device=device), 1)

    beams = [([], 0.0, False)]
    for i in range(num_steps):
        completed = [b for b in beams if b[2]]
        active = [b for b in beams if not b[2]]
        if not active:
            break

        j = i + 1
        tgt = torch.full((len(active), j), int(model.pad_id), dtype=torch.long, device=device)
        tgt[:, 0] = int(model.bos_id)
        for bi, (tokens, _, _) in enumerate(active):
            if tokens:
                tgt[bi, 1:] = torch.tensor(tokens, dtype=torch.long, device=device)

        mem = memory.expand(len(active), -1, -1)
        pq = pos_queries[:, i:j].expand(len(active), -1, -1)
        out = model.decode(
            tgt,
            mem,
            mask[:j, :j],
            tgt_query=pq,
            tgt_query_mask=mask[i:j, :j],
        )
        logp = F.log_softmax(model.head(out).squeeze(1), dim=-1)
        topv, topi = torch.topk(logp, k=min(beam_width, logp.shape[-1]), dim=-1)

        proposals = list(completed)
        for bi, (tokens, score, _) in enumerate(active):
            for lp, tid in zip(topv[bi].tolist(), topi[bi].tolist()):
                tid = int(tid)
                proposals.append((tokens + [tid], score + float(lp), tid == int(model.eos_id)))
        proposals.sort(key=lambda z: z[1], reverse=True)
        beams = proposals[:beam_width]
        if all(x[2] for x in beams):
            break

    return unique_texts(model, [(t, s) for t, s, _ in beams], beam_width)


@torch.inference_mode()
def final_logit_lattice(model, image_tensor, beam_width=20):
    logits = model(model.tokenizer, image_tensor)[0]
    logp = F.log_softmax(logits, dim=-1)
    beams = [([], 0.0, False)]
    for i in range(logp.shape[0]):
        vals, ids = torch.topk(logp[i], k=min(beam_width, logp.shape[-1]))
        proposals = []
        for tokens, score, ended in beams:
            if ended:
                proposals.append((tokens, score, True))
                continue
            for lp, tid in zip(vals.tolist(), ids.tolist()):
                tid = int(tid)
                proposals.append((tokens + [tid], score + float(lp), tid == int(model.eos_id)))
        proposals.sort(key=lambda z: z[1], reverse=True)
        beams = proposals[:beam_width]
        if all(x[2] for x in beams):
            break
    return unique_texts(model, [(t, s) for t, s, _ in beams], beam_width)


def rank_of(gt, cands):
    g = gt.casefold()
    for i, x in enumerate(cands, 1):
        if x["text"].casefold() == g:
            return i
    return None


def summarize(rows, rank_key):
    errors = [r for r in rows if not r["v1_exact"]]
    out = {"eligible_gt_rows": len(rows), "v1_errors": len(errors)}
    for k in KS:
        out[f"top{k}_all"] = sum(r[rank_key] is not None and r[rank_key] <= k for r in rows)
        out[f"top{k}_errors_recoverable"] = sum(r[rank_key] is not None and r[rank_key] <= k for r in errors)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact-root", type=Path)
    ap.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    ap.add_argument("--parseq-source", type=Path)
    ap.add_argument("--parseq-checkpoint", type=Path)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-beam", type=int, default=20)
    ap.add_argument("--limit-unique-crops", type=int, default=0)
    a = ap.parse_args()

    compact = (a.compact_root or discover_compact()).resolve()
    source = (a.parseq_source or discover_parseq_source()).resolve()
    checkpoint = (a.parseq_checkpoint or discover_parseq_checkpoint()).resolve()
    audit = a.audit.resolve()

    print("[paths]")
    print(" compact    =", compact)
    print(" source     =", source)
    print(" checkpoint =", checkpoint)
    print(" audit      =", audit)

    rows = load_population(compact, audit)
    by_crop = defaultdict(list)
    for r in rows:
        by_crop[(r["sample_id"], r["candidate_id"], r["crop_path"])].append(r)

    if a.limit_unique_crops:
        keys = list(by_crop)[:a.limit_unique_crops]
        by_crop = {k: by_crop[k] for k in keys}
        rows = [r for k in keys for r in by_crop[k]]

    print(f"[population] eligible GT rows={len(rows)} unique frozen V1 crops={len(by_crop)} V1 exact={sum(r['v1_exact'] for r in rows)}")

    sys.path.insert(0, str(source))
    from strhub.models.utils import create_model
    from strhub.data.module import SceneTextDataModule

    device = torch.device(a.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Run on PAPER GPU.")
    torch.cuda.set_device(device)

    # The official PARSeq v1.0.0 release .pt is a raw state_dict for the
    # inner PARSeq model, NOT a Lightning checkpoint. This mirrors upstream
    # create_model(..., pretrained=True) but uses our pinned local file.
    system = create_model("parseq", pretrained=False)
    state = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(state, dict) or "pytorch-lightning_version" in state:
        raise RuntimeError("Unexpected PARSeq checkpoint format; expected official raw state_dict")
    system.model.load_state_dict(state, strict=True)
    system = system.eval().to(device)

    # Candidate generation operates on the inner decoder but uses the exact
    # tokenizer/configuration owned by the official PARSeq system wrapper.
    model = system.model
    model.tokenizer = system.tokenizer
    model.bos_id = int(system.tokenizer.bos_id)
    model.eos_id = int(system.tokenizer.eos_id)
    model.pad_id = int(system.tokenizer.pad_id)
    transform = SceneTextDataModule.get_transform(system.hparams.img_size)

    cache = {}
    for n, (key, assoc) in enumerate(by_crop.items(), 1):
        sid, cid, crop_path = key
        with Image.open(crop_path) as im:
            x = transform(im.convert("RGB")).unsqueeze(0).to(device)
        ar = ar_beam(model, x, a.max_beam)
        final = final_logit_lattice(model, x, a.max_beam)
        cache[(sid, cid)] = (ar, final)
        print(f"[{n}/{len(by_crop)}] sample={sid} cid={cid} ar={len(ar)} final={len(final)}", flush=True)

    details = []
    for r in rows:
        ar, final = cache[(r["sample_id"], r["candidate_id"])]
        d = dict(r)
        d["ar_rank"] = rank_of(r["gt_text"], ar)
        d["final_rank"] = rank_of(r["gt_text"], final)
        d["ar_candidates"] = ar
        d["final_candidates"] = final
        details.append(d)

    report = {
        "classification": "DIAGNOSTIC_ONLY_TRAIN_GT_ORACLE_CANDIDATE_COVERAGE",
        "scope": {
            "split": "TRAIN",
            "gt_rows_evaluated": len(details),
            "unique_frozen_v1_crops": len(by_crop),
            "test_loaded": False,
        },
        "candidate_generation": {
            "ar_beam": "canonical forward autoregressive beam before iterative refinement",
            "final_logit_lattice": "k-best strings from final post-refinement logits",
            "max_beam": a.max_beam,
            "context_used": False,
            "training": False,
            "crop_changes": False,
        },
        "controls": {
            "v1_exact_on_eligible_rows": sum(x["v1_exact"] for x in details),
            "v1_errors_on_eligible_rows": sum(not x["v1_exact"] for x in details),
        },
        "ar_beam": summarize(details, "ar_rank"),
        "final_logit_lattice": summarize(details, "final_rank"),
        "details": details,
    }

    a.output_dir.mkdir(parents=True, exist_ok=True)
    out = a.output_dir / "results.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== PARSEQ CANDIDATE COVERAGE ===")
    print(f"eligible GT rows = {len(details)}")
    print(f"V1 exact/errors  = {report['controls']['v1_exact_on_eligible_rows']}/{report['controls']['v1_errors_on_eligible_rows']}")
    print("\nsource                 top1 top3 top5 top10 top20")
    for name, block in (("AR beam", report["ar_beam"]), ("final logits", report["final_logit_lattice"])):
        vals = [block[f"top{k}_all"] for k in KS]
        print(f"{name:20s} " + " ".join(f"{v:5d}" for v in vals))
    print("\nRECOVERABLE AMONG CURRENT V1 ERRORS")
    for name, block in (("AR beam", report["ar_beam"]), ("final logits", report["final_logit_lattice"])):
        vals = [block[f"top{k}_errors_recoverable"] for k in KS]
        print(f"{name:20s} " + " ".join(f"{v:5d}" for v in vals))
    print("JSON", out)


if __name__ == "__main__":
    main()
