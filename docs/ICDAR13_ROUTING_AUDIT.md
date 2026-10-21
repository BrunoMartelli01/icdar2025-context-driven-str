# ICDAR13 routing audit and persistent comparison

## Historical result recovered on 2026-09-10

`PAPER_AMBIGUITIES.md`, R-E2E-COMP-001, records a completed official TEST-233
comparison of Algorithm1 and BALANCED_ANCHOR_MEAN:

| Diagnostic | Algorithm1 | Balanced mean |
|---|---:|---:|
| GT words with proposal IoU >= .50 | 174/1095 | 528/1095 |
| PARSeq exact on best geometric crop | 102/1095 | 453/1095 |
| Pre-fallback selected string belongs to image GT | 71/233 | 152/233 |
| Conditional fallback images | 233/233 | 233/233 |
| Final highest-score string belongs to image GT | 207/233 | 207/233 |

207/233 = **88.841%**. Both branches used DeepSolo for every final prediction.
All raw DeepSolo strings instead gave 861/1095 one-to-one, per-image casefold
matches: recall 78.63%, precision 83.92%, F1 81.19%. The 41/66 = 62.12%
number cited later belongs to a different, first-ten-images diagnostic. It is
not a remeasurement of either full-TEST metric. The earlier conversational claim
that only first-ten DeepSolo results had ever existed was incorrect.

The historical full JSON (`/tmp/icdar13-e2e-faithful-vs-balanced-v1.json`) and
its inference shards are currently absent from the searched run/artifact/tmp
directories. The recorded SHA256 is
`e6db8027e1d0f6ec3bf5e3b574c431ad1779857330825cb14bf766f505747eee`.
These numbers are recovered from the experiment record; they have not been
recomputed from the missing original bytes during this audit.

The evaluator's `_aggregate_final` implements image membership and the separate
per-image word multiset precisely this way. Paper Table 5 gives 88.8 for the full
R50 pipeline, while the surrounding text reports 168/229 images avoiding
fallback. Agreement after rounding does not resolve E-001, DS-002, or D-002.
The full official TEST manifest used locally has 233 images and 1095 care words.

## Frozen comparison to execute on PAPER

`scripts/evaluate_icdar13_routing_suite.py` reuses the historical proposal, PARSeq,
original caption and selection functions. No localization or selector weights
are changed. The missing TRAIN balanced report is not silently recreated: its
historical hash/status are recorded, while the present implementation hashes
and the verified surviving v1 calibration define this fresh execution.

Predeclared branches:

1. Algorithm1 + repository caption/scorer (historical control).
2. Practical v1 + repository caption/scorer.
3. Balanced mean + repository caption/scorer (historical control).
4. Practical v1 + exact-length local context, original caption.
5. Balanced mean + exact-length local context, original caption.
6. Practical v1 + exact-length local context, anti-repetition caption.
7. Balanced mean + exact-length local context, anti-repetition caption.
8. DeepSolo standalone shared control.

The local gate is the TRAIN-tested T3-only maximum paired window score. It does
not add T1 to the local gate; the historical scorer still compares T1 and T3.
Consequently the comparison to historical routing includes this explicit routing
change. Local-window scores use .6/.4 and tau .8; no TEST optimization occurs.
Balanced proposals are frozen using the original caption before the local
context variants, so caption changes do not alter proposal selection.

One segmentation result per image feeds all localizers; identical PARSeq boxes
are evaluated once. Each caption profile runs once per image. Models are loaded
sequentially. All local scoring runs on CPU. DeepSolo runs once on every image
to provide a standalone/counterfactual control, independent of per-branch logical
fallback counts. The report distinguishes these counts explicitly.

For each branch the report saves image-membership rate, pre-fallback membership,
fallback/no-fallback counts, accepted correct/incorrect counts, accepted words
also found in raw DeepSolo, gains/losses against DeepSolo-only, C median/max,
and word multiset precision/recall/F1. The word population is all raw DeepSolo
strings on fallback and one selected PARSeq string on acceptance, exactly as in
the historical diagnostic; it must not be treated as an official spatial E2E
score. Best-geometric-crop PARSeq results remain post-hoc diagnostics.

## Run and storage

From the repository on PAPER:

```bash
bash scripts/run_icdar13_routing_suite_host.sh /extra1/bmartelli --check
bash scripts/run_icdar13_routing_suite_host.sh /extra1/bmartelli
```

The first command verifies frozen inputs and all TEST RGB checksums without GPU
inference or experiment writes. The second runs in foreground with visible
progress, validates model assets, then executes the suite.

Default output: `context-str-runs/icdar13-routing-suite-v1/results.json`, relative
to the supplied storage root. All shards are persistent under its `artifacts/`.
Rerun the same command after interruption; completed stages are SHA-verified and
skipped. A source/input change invalidates the run contract and fails explicitly.
Use a different `CONTEXT_STR_ROUTING_SUITE_ROOT` for an intentional separate run.
No crop images are stored: retained metadata refers to original RGB coordinates.
No legacy `/tmp` reports are required. No dependencies are installed.

The native model execution requires manual GPU qualification; CPU fixture tests
and the real read-only preflight do not certify the new complete GPU suite.
