#!/usr/bin/env python3
"""
DIAGNOSTIC_ONLY_ORACLE: structural upper bound on frozen TRAIN229 V1 OCR crops.

No LLM, captions, geometry thresholds, edit distance, dictionary or semantic model.
GT is used only by the oracle scorer/search.

Conditions:
  V1
  KEEP_SPLIT
  KEEP_MERGE_CONCAT
  KEEP_MERGE_CONCAT_SPLIT
  KEEP_MERGE_OVERLAP_SPLIT

The structural oracle preserves characters. SPLIT may cut only at character
boundaries. MERGE concatenates selected crops in frozen reading order.
MERGE_OVERLAP additionally permits exact raw suffix-prefix deduplication.

The main number is a feasible exact-match upper bound with every crop consumed
exactly once by one structural operation. It is intentionally optimistic because
there is no geometric plausibility filter.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


CONDITIONS = (
    "KEEP_SPLIT",
    "KEEP_MERGE_CONCAT",
    "KEEP_MERGE_CONCAT_SPLIT",
    "KEEP_MERGE_OVERLAP_SPLIT",
)


def lexical_counts(predicted, expected):
    pred = Counter(" ".join(predicted).casefold().split())
    gt = Counter(x.casefold() for x in expected)
    matches = sum((pred & gt).values())
    return {
        "matches": matches,
        "predicted": sum(pred.values()),
        "gt": sum(gt.values()),
        "surplus": sum((pred - gt).values()),
        "missing": sum((gt - pred).values()),
    }


def exact_overlap_merge(parts):
    if not parts:
        return ""
    out = parts[0]
    for nxt in parts[1:]:
        best = 0
        lim = min(len(out), len(nxt))
        for k in range(1, lim + 1):
            if out[-k:] == nxt[:k]:
                best = k
        out += nxt[best:]
    return out


def cap_add(a, b, caps):
    return tuple(min(caps[i], a[i] + b[i]) for i in range(len(caps)))


def dominates(a, b):
    return a != b and all(x >= y for x, y in zip(a, b))


def pareto(states):
    states = set(states)
    ordered = sorted(states, key=lambda s: (sum(s), s), reverse=True)
    kept = []
    for s in ordered:
        if any(all(k[i] >= s[i] for i in range(len(s))) for k in kept):
            continue
        kept.append(s)
    return set(kept)


def fixed_token_state(text, words, caps):
    counts = [0] * len(words)
    index = {w: i for i, w in enumerate(words)}
    for tok in text.casefold().split():
        if tok in index:
            j = index[tok]
            if counts[j] < caps[j]:
                counts[j] += 1
    return {tuple(counts)}


def split_chunk_states(chunk, words, caps):
    """All Pareto-relevant GT-count states obtainable by arbitrary char splits.
    Unmatched characters are preserved but irrelevant to exact-match count.
    """
    chunk = chunk.casefold()
    zero = (0,) * len(words)
    dp = {0: {zero}}
    for pos in range(len(chunk) + 1):
        if pos not in dp:
            continue
        here = pareto(dp[pos])
        dp[pos] = here
        if pos == len(chunk):
            continue

        # Preserve an unmatched character. Consecutive unmatched chars can form
        # one output token; token count is intentionally not optimized here.
        dp.setdefault(pos + 1, set()).update(here)

        for j, w in enumerate(words):
            if not w or not chunk.startswith(w, pos):
                continue
            end = pos + len(w)
            for st in here:
                if st[j] >= caps[j]:
                    continue
                ns = list(st)
                ns[j] += 1
                dp.setdefault(end, set()).add(tuple(ns))

    return pareto(dp.get(len(chunk), {zero}))


def split_states(text, words, caps):
    # Existing OCR whitespace is already a valid token boundary.
    chunks = text.split()
    states = {(0,) * len(words)}
    for chunk in chunks:
        local = split_chunk_states(chunk, words, caps)
        states = pareto(
            cap_add(a, b, caps)
            for a in states
            for b in local
        )
    return states


def subset_parts(blocks, mask):
    return [blocks[i]["text"] for i in range(len(blocks)) if mask & (1 << i)]


def operation_states(blocks, mask, words, caps, condition):
    parts = subset_parts(blocks, mask)
    if condition == "KEEP_SPLIT":
        if mask & (mask - 1):
            return set()
        return split_states(parts[0], words, caps)

    concat = "".join(parts)
    if condition == "KEEP_MERGE_CONCAT":
        return fixed_token_state(concat, words, caps)

    if condition == "KEEP_MERGE_CONCAT_SPLIT":
        return split_states(concat, words, caps)

    if condition == "KEEP_MERGE_OVERLAP_SPLIT":
        states = set(split_states(concat, words, caps))
        overlap = exact_overlap_merge(parts)
        states.update(split_states(overlap, words, caps))
        return pareto(states)

    raise ValueError(condition)


def feasible_oracle(blocks, gt_words, condition):
    """Exact partition DP. Every crop is consumed exactly once.
    No geometric restrictions; subsets may be arbitrary.
    """
    gt_counter = Counter(w.casefold() for w in gt_words)
    words = tuple(sorted(gt_counter))
    caps = tuple(gt_counter[w] for w in words)
    n = len(blocks)
    full = (1 << n) - 1
    zero = (0,) * len(words)

    op_cache = {}
    if condition == "KEEP_SPLIT":
        masks = [1 << i for i in range(n)]
    else:
        masks = list(range(1, full + 1))

    for mask in masks:
        op_cache[mask] = operation_states(blocks, mask, words, caps, condition)

    dp = {0: {zero}}
    for used in range(full + 1):
        states = dp.get(used)
        if not states:
            continue
        states = pareto(states)
        dp[used] = states
        if used == full:
            continue
        first = next(i for i in range(n) if not (used & (1 << i)))
        available = full ^ used

        sub = available
        while sub:
            if sub & (1 << first) and sub in op_cache:
                local = op_cache[sub]
                if local:
                    new_used = used | sub
                    bucket = dp.setdefault(new_used, set())
                    for a in states:
                        for b in local:
                            bucket.add(cap_add(a, b, caps))
                    # Keep growth bounded aggressively; crop count <= 10.
                    if len(bucket) > 256:
                        dp[new_used] = pareto(bucket)
            sub = (sub - 1) & available

    finals = pareto(dp.get(full, {zero}))
    best = max(finals, key=lambda s: (sum(s), s))
    matched = sum(best)
    matched_words = {
        words[i]: best[i] for i in range(len(words)) if best[i]
    }
    return matched, matched_words


def candidate_coverage(blocks, gt_words, condition):
    """Very optimistic per-occurrence coverage, ignoring crop conflicts."""
    gt_counter = Counter(w.casefold() for w in gt_words)
    words = tuple(sorted(gt_counter))
    caps = tuple(gt_counter[w] for w in words)
    n = len(blocks)
    full = (1 << n) - 1
    max_seen = [0] * len(words)

    masks = [1 << i for i in range(n)] if condition == "KEEP_SPLIT" else range(1, full + 1)
    for mask in masks:
        for st in operation_states(blocks, mask, words, caps, condition):
            for i, x in enumerate(st):
                if x > max_seen[i]:
                    max_seen[i] = x

    # If a word is producible k times inside one operation, count up to GT multiplicity.
    return sum(min(caps[i], max_seen[i]) for i in range(len(words)))


def load_data(source_root, audit_path):
    source_root = Path(source_root)
    inputs_path = source_root / "inputs.json"
    if not inputs_path.exists():
        raise FileNotFoundError(f"missing {inputs_path}")
    inputs = json.loads(inputs_path.read_text())
    samples = inputs["samples"]

    audit = json.loads(Path(audit_path).read_text())
    rows = audit["per_word"]

    gt = {}
    for r in rows:
        gt.setdefault(str(r["sample_id"]), []).append(r["gt_text"])

    if len(samples) != 229:
        raise RuntimeError(f"expected 229 TRAIN images, found {len(samples)}")
    if sum(len(s["blocks"]) for s in samples) != 1203:
        raise RuntimeError("expected frozen 1203 V1 crops")
    if sum(len(v) for v in gt.values()) != 849:
        raise RuntimeError("expected 849 TRAIN GT words")

    return samples, gt


def run(source_root, audit_path, output_dir):
    samples, gt = load_data(source_root, audit_path)

    baseline_matches = 0
    baseline_pred = 0
    per_image = []

    totals = {c: 0 for c in CONDITIONS}
    coverage = {c: 0 for c in CONDITIONS}

    for idx, sample in enumerate(samples, 1):
        sid = str(sample["sample_id"])
        blocks = sample["blocks"]
        expected = gt[sid]

        lc = lexical_counts([b["text"] for b in blocks], expected)
        baseline_matches += lc["matches"]
        baseline_pred += lc["predicted"]

        row = {
            "sample_id": sid,
            "v1_matches": lc["matches"],
            "conditions": {},
        }

        for condition in CONDITIONS:
            m, matched_words = feasible_oracle(blocks, expected, condition)
            cov = candidate_coverage(blocks, expected, condition)
            totals[condition] += m
            coverage[condition] += cov
            row["conditions"][condition] = {
                "feasible_matches": m,
                "candidate_coverage_matches": cov,
                "matched_gt_casefold": matched_words,
            }

        per_image.append(row)
        if idx % 25 == 0 or idx == len(samples):
            print(f"[oracle] {idx}/{len(samples)} images", flush=True)

    if baseline_matches != 503:
        raise RuntimeError(
            f"V1 control mismatch: expected 503 image-level lexical matches, got {baseline_matches}"
        )
    if baseline_pred != 1203:
        raise RuntimeError(
            f"V1 token control mismatch: expected 1203 predicted tokens, got {baseline_pred}"
        )

    summary = {
        "V1": {
            "image_lexical_exact_matches": baseline_matches,
            "gt_words": 849,
            "predicted_tokens": baseline_pred,
            "recall": baseline_matches / 849,
            "gain_vs_v1": 0,
        }
    }
    for condition in CONDITIONS:
        summary[condition] = {
            "feasible_oracle_exact_matches": totals[condition],
            "candidate_coverage_matches": coverage[condition],
            "gt_words": 849,
            "recall_upper_bound": totals[condition] / 849,
            "gain_vs_v1": totals[condition] - baseline_matches,
        }

    result = {
        "classification": "DIAGNOSTIC_ONLY_ORACLE",
        "scope": {"split": "TRAIN", "images": 229, "gt_words": 849, "crops": 1203},
        "rules": {
            "gt_used_only_for_oracle_search_and_scoring": True,
            "caption": False,
            "llm": False,
            "geometry_filter": False,
            "edit_correction": False,
            "dictionary": False,
            "merge_subset": "any crop subset, frozen reading order",
            "split": "arbitrary character boundaries; characters preserved",
            "overlap": "exact raw suffix-prefix only",
            "every_crop_consumed_exactly_once_in_feasible_oracle": True,
        },
        "summary": summary,
        "per_image": per_image,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "results.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== STRUCTURAL ORACLE ===")
    print(f"V1                         {baseline_matches}/849")
    for c in CONDITIONS:
        s = summary[c]
        print(
            f"{c:28s} feasible={s['feasible_oracle_exact_matches']:3d}/849 "
            f"gain={s['gain_vs_v1']:+4d} "
            f"coverage={s['candidate_coverage_matches']:3d}/849"
        )
    print("JSON", out)


def self_test():
    def solve(texts, gt, cond):
        blocks = [{"text": x} for x in texts]
        return feasible_oracle(blocks, gt, cond)[0]

    assert solve(["F", "lu", "or", "ide"], ["Fluoride"], "KEEP_MERGE_CONCAT") == 1
    assert solve(["USEONLY"], ["USE", "ONLY"], "KEEP_SPLIT") == 2
    assert solve(["ABC", "BCD"], ["ABCD"], "KEEP_MERGE_OVERLAP_SPLIT") == 1
    assert solve(["ABC", "BXD"], ["ABCD"], "KEEP_MERGE_OVERLAP_SPLIT") == 0
    assert solve(["cat"], ["cat", "cat"], "KEEP_SPLIT") == 1
    assert solve(["cat", "cat"], ["cat", "cat"], "KEEP_SPLIT") == 2
    # Same crop cannot be reused to satisfy both unrelated GT occurrences.
    assert solve(["ABC"], ["A", "ABC"], "KEEP_SPLIT") == 1
    # Exact overlap is allowed, fuzzy character repair is not.
    assert exact_overlap_merge(["ABC", "BCD"]) == "ABCD"
    assert exact_overlap_merge(["ABC", "BXD"]) == "ABCBXD"
    print("SELF-TEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root")
    ap.add_argument("--audit")
    ap.add_argument("--output-dir")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        self_test()
        return
    if not (a.source_root and a.audit and a.output_dir):
        ap.error("--source-root, --audit and --output-dir are required unless --self-test")
    run(a.source_root, a.audit, a.output_dir)


if __name__ == "__main__":
    main()
