"""Independent order-sensitive dictionary lookup, policy and full word-count audit."""

import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path

from scripts.evaluate_deepsolo_icdar_lexicon import read, sha, save


def edit_at_most_one(a, b):
    if a == b:
        return 0
    if abs(len(a)-len(b)) > 1:
        return None
    if len(a) == len(b):
        return 1 if sum(x != y for x, y in zip(a, b)) == 1 else None
    if len(a) > len(b):
        a, b = b, a
    i = j = skips = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
        else:
            skips += 1
        j += 1
        if skips > 1:
            return None
    return 1


class IndependentLexicon:
    def __init__(self, vocabulary, pairs):
        self.words = [s.strip().upper() for s in Path(vocabulary).read_text().splitlines()]
        self.pairs = {}
        for line in Path(pairs).read_text().splitlines():
            token, _, canonical = line.strip().partition(" ")
            self.pairs[token.upper()] = canonical
        self.exact = {}
        self.lengths = {}
        for i, word in enumerate(self.words):
            self.exact.setdefault(word, i)
            self.lengths.setdefault(len(word), []).append(i)

    @lru_cache(maxsize=None)
    def match(self, text):
        q = text.upper()
        if q in self.exact:
            i = self.exact[q]
            return dict(text=self.pairs[q], lexicon_word=q, distance=0, lexicon_index=i), 1
        # A distance<=1 match necessarily has length difference<=1. Inspect
        # candidates in original file order using an independent character scan.
        indices = sorted(i for n in (len(q)-1, len(q), len(q)+1) for i in self.lengths.get(n, []))
        hits = [i for i in indices if edit_at_most_one(q, self.words[i]) == 1]
        if not hits:
            return None, 0
        i = hits[0]
        word = self.words[i]
        return dict(text=self.pairs[word], lexicon_word=word, distance=1, lexicon_index=i), len(hits)


def counts(rows, gt):
    normalized = lambda s: "".join(c for c in s.casefold() if c.isalnum())
    remaining = Counter(normalized(g["text"]) for g in gt if normalized(g["text"]))
    tp = n = ignored = 0
    for p in rows:
        if p["ignored"]:
            ignored += 1
            continue
        n += 1
        word = normalized(p["text"])
        if p.get("invalid_envelope") is None and word and remaining[word]:
            tp += 1
            remaining[word] -= 1
    return dict(tp=tp, fp=n-tp, fn=len(gt)-tp, detected_words=n, gt_words=len(gt),
                all_outputs=len(rows), ignored_outputs=ignored)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    c = read(args.config)
    root = Path(c["study_root"])
    checks = Counter()
    cache = {}
    def lexicon(base, kind, sid):
        identity = sid.split(":")[-1]
        if kind == "Generic":
            paths = base/"GenericVocabulary_new.txt", base/"GenericVocabulary_pair_list.txt"
        elif kind == "Weak":
            prefix = "ch2" if base.name == "ic13" else "ch4"
            paths = base/(prefix+"_test_vocabulary_new.txt"), base/(prefix+"_test_vocabulary_pair_list.txt")
        else:
            paths = base/"new_strong_lexicon"/("new_voc_img_"+identity+".txt"), base/"new_strong_lexicon"/("pair_voc_img_"+identity+".txt")
        key = tuple(str(p) for p in paths)
        if key not in cache:
            cache[key] = IndependentLexicon(*paths)
        return cache[key]
    simple = read(root/"results.json")
    cases = [(ds, root, v["reports"], Path(c["lexicon_root"])) for ds,v in simple["datasets"].items()]
    special = root/"ic13_dataset_v1"
    cc = read(Path(args.config).parent/"deepsolo_ic13_lexicons_v1.json")
    cases.append(("icdar13", special, read(special/"results.json")["reports"], Path(cc["ic13_lexicon_root"])))
    # Independent replay of all fixed neural admission, vocabulary, transcription
    # and ignore decisions; full ground-truth populations stay identical.
    for ds, directory, reports, lexroot in cases:
        scenes = read(root/(ds+"_test_observations.json"))
        for name, report in reports.items():
            filename = (ds+"__" if directory == root else "") + name + "_per_image.json"
            path = directory/filename
            assert sha(path) == report["per_image_sha256"]
            saved = read(path)
            assert set(saved) == set(scenes)
            aggregate = Counter()
            for sid, scene in scenes.items():
                p = report["neural"]
                original = sorted((v for v in scene["predictions"] if v["score"] >= p["score_threshold"] and
                                   v["recognition_confidence"] >= p["recognition_threshold"]),
                                  key=lambda v: (-v["score"], v["instance_id"]))
                final = []
                decisions = []
                kind = report["setting"].split("_")[0]
                for v in original:
                    if kind == "None":
                        final.append(v)
                        continue
                    model = lexicon(lexroot, kind, sid)
                    match, _ = model.match(v["text"])
                    accepted = match is not None or report["unmatched_policy"] == "keep_unmatched"
                    text = match["text"] if match is not None else v["text"]
                    decisions.append(dict(instance_id=v["instance_id"], original_text=v["text"],
                        accepted=accepted, corrected_text=text if accepted else None,
                        match=match, unmatched_policy=report["unmatched_policy"]))
                    if accepted:
                        final.append(dict(v, text=text))
                    checks["independent_dictionary_decisions"] += 1
                assert final == saved[sid]["predictions"]
                assert decisions == saved[sid]["decisions"]
                assert scene["ground_truth"] == saved[sid]["ground_truth"]
                values = counts(final, scene["ground_truth"])
                assert values == saved[sid]["counts"]
                aggregate.update(values)
                checks["independent_image_count_replays"] += 1
            m = report["metrics"]
            for key, value in aggregate.items():
                assert value == m[key], (ds, name, key)
            assert 2*aggregate["tp"]/(aggregate["detected_words"]+aggregate["gt_words"]) == m["f1"]
            assert m["deepsolo_images"] == m["total_images"] == len(scenes)
            checks["complete_metric_replays"] += 1
    # Calibration optima are replayed from their saved full finite grids; the
    # actual dictionary decisions above are independently implemented.
    for directory, field in [(root,"choices"),(special,"choice")]:
        frozen = read(directory/"frozen_choices.json")
        assert frozen["test_used"] is False
        grids = read(directory/"calibration_grid.json")
        groups = grids.items() if isinstance(grids,dict) else [("icdar13",grids)]
        for ds, grid in groups:
            scenes = read(root/(ds+"_validation_observations.json"))
            lexroot = Path(c["lexicon_root"]) if directory == root else Path(cc["ic13_lexicon_root"])
            for point in grid:
                aggregate = Counter()
                for sid, scene in scenes.items():
                    p = point["neural"]
                    admitted = sorted((v for v in scene["predictions"] if v["score"] >= p["score_threshold"] and
                                       v["recognition_confidence"] >= p["recognition_threshold"]),
                                      key=lambda v:(-v["score"],v["instance_id"]))
                    final=[]
                    model=lexicon(lexroot,"Generic",sid)
                    for v in admitted:
                        match,_=model.match(v["text"])
                        if match is not None:
                            final.append(dict(v,text=match["text"]))
                        elif point["unmatched_policy"]=="keep_unmatched":
                            final.append(v)
                    aggregate.update(counts(final,scene["ground_truth"]))
                    checks["independent_calibration_image_replays"]+=1
                for name,value in aggregate.items():
                    assert value==point["metrics"][name]
                assert 2*aggregate["tp"]/(aggregate["detected_words"]+aggregate["gt_words"])==point["metrics"]["f1"]
                checks["independent_calibration_grid_replays"]+=1
            names = list(read(c["protocol_path"])["confidence_candidates"][ds])
            def key(v):
                m=v["metrics"]
                return (Fraction(2*m["tp"],m["detected_words"]+m["gt_words"]),m["tp"],
                        v["unmatched_policy"]=="keep_unmatched",-names.index(v["confidence_name"]))
            chosen=max(grid,key=key)
            assert chosen == (frozen[field][ds] if field=="choices" else frozen[field])
            checks["frozen_calibration_optima"] += 1
    result=dict(status="PASS",mode="approximate",claim_label="approximate reproduction",
                checks=dict(checks),independent_lookup="Length-bounded character scan, original-file first ties, independent pair-list parser; no RapidFuzz used by the auditor.",
                results_sha256=sha(root/"results.json"),ic13_results_sha256=sha(special/"results.json"),
                code_sha256=sha(__file__),scope="Full primary/source vocabulary cases; guarded follow-up has separate focused boundary tests and is not a promoted profile.")
    save(args.output,result)
    print(result,flush=True)


if __name__ == "__main__":
    main()
