# Frozen OCR: completion, structural selection, and byte denoising

2026-10-23. **Approximate reproduction / exploratory component experiments.**
The user authorized implementation, execution, and result analysis of three
text-reconstruction proposals. U-Net, segmentation masks, localized RGB crops,
PARSeq weights/decoding, and the strict recognition pipeline remain fixed.
This is not paper Table 5 or a complete BLIP-2/MPNet/DeepSolo evaluation.

Implementation: `recognition/text_reconstruction.py`,
`scripts/run_text_reconstruction_tests.py`,
`scripts/replay_qwen_insertion_controls.py`,
`scripts/audit_text_reconstruction_tests.py`, and
`scripts/analyze_text_reconstruction_tests.py`, with
`tests/recognition/test_text_reconstruction.py`.
Configurations: `configs/approximate/text_reconstruction_tests_v1.json` and
`configs/approximate/qwen_insertion_replay_v1.json`.

## Scientific basis and earlier evidence

The authoritative local paper sections 3.3–3.5 and Algorithm 1, genuine NeuSpell
word-head implementation, historical Qwen fragment reconstruction/structural
selection, and the October 22 M/K saved outputs were inspected before changes.
The paper does not specify the new correction/training mechanisms.

Earlier Total-Text development showed a selective insertion-only NeuSpell gain
384 to 396, recovering 14 words and damaging 2. Broad correction was negative.
The historical ICDAR13 structural oracle reached 627 lexical matches from 503,
but used unrestricted GT-selected operations; Qwen selection reached only 450.
The 80-condition Qwen battery has now completed: its original conditions all
remain NO_GO, maximum 503 versus 502. The earlier summary's pending status is
historical, not the current artifact status.

Primary inspiration:

- [Tokenization Repair in the Presence of Spelling Errors, CoNLL 2021](https://aclanthology.org/2021.conll-1.22/)
  supports studying missing/spurious boundaries separately from spelling.
- [ByT5, TACL 2022](https://aclanthology.org/2022.tacl-1.17/) supplies a pretrained
  byte-level encoder-decoder, not an OCR-specific pretrained corrector.
- [Advancing Post-OCR Correction, ACL 2024](https://aclanthology.org/2024.findings-acl.361/)
  motivates synthetic OCR corruption. Our span/group noise is an adaptation,
  not an implementation or reproduction of that paper's corruption generator.

## Fixed data and evidence boundaries

The source is the completed
`occlusion-four-tests-20261022-v1/full` run. Its protocol, final export manifest,
complete OCR JSONL, and annotation hash inventories are pinned in the new config.
Every selected export and annotation is checksum-verified. Only the original
M/K baseline observations enter the three main tests; no alternative RGB crops
or new OCR readings are introduced.

Reuse 1030 fit, 100 calibration, 125 evaluation, and 300 TEST images. All text
priors, supervised labels, and real/synthetic denoiser pairs use fit annotations
only. Calibration chooses policies/checkpoints. New evaluation/TEST transcripts
are parsed only after choices are frozen; checksum verification is not parsing
or supplying annotations to inference.

These historical cohorts were repeatedly explored and are not independent
validation. The source segmenter already saw the calibration images. Known
pixel-identical TEST578/TRAIN212 and TEST656/TRAIN81 stay in primary results;
a separate 298-image sensitivity excludes exactly those TEST IDs. Other
cross-role pixel identity fails preflight.

## Three experiments

1. **Completion without replacing observed letters.** Genuine frozen NeuSpell
   uses the published trained word head/vocabulary. Keep the earlier OOV,
   probability >=0.8, source length >=4, at-most-two-insertion control. A separate
   long-completion control permits up to eight insertions from source length3,
   with the same probability/OOV restrictions. Existing eligibility preserves
   numbers, punctuation, and internal mixed case. No caption is added.
2. **Learned structural selection.** Build horizontal, geometry-limited
   single-observation splits and two/three-observation concatenation or exact
   overlap alternatives, optionally followed by splitting. The external
   NeuSpell vocabulary plus fit-only words permits boundaries; fit-only character
   trigrams and word counts are features. A balanced HistGradientBoosting
   classifier fits 150 iterations on actual per-image F1-improving operations
   that do not lose lexical matches. Greedy conflict resolution consumes every
   source exactly once. Seven fixed thresholds include KEEP; calibration chooses
   most correct words, then F1, under at most two lost lexical matches. A fixed
   0.5 raw arm is also reported, without TEST selection. The separate integer
   program finds a feasible optimum of this actual bounded candidate pool under
   one-use and GT multiplicity constraints. It is a GT oracle diagnostic only.
3. **Pretrained byte denoising.** Fine-tune actual `google/byt5-small`, pinned
   revision `68377bdc18a2ffec8a0533fef03b1c513a4dd49d`, all 299,637,760
   parameters. Three epochs over a deterministic maximum18,000 distinct
   source/target/corruption-type records, batch16, AdamW1e-4/decay0.01,
   gradient clipping1. Real OCR groups receive
   geometry-associated fit targets; these targets can be noisy. Synthetic pairs
   include identity, contiguous span deletion, split/overlapping fragments, and
   merged word boundaries. Fit-derived examples exceeding128 input/96 target
   byte limits are counted and excluded from training, never from evaluation.
   The best raw calibration endpoint is selected; all three epochs execute.
   Greedy inference reports raw bounded corrections and a matched policy retaining
   every observed letter in order while permitting changed spaces. Both operate
   on the original observations and on the calibrated structural outputs, with
   the same checkpoint. A separate calibrated deployment arm may choose KEEP.

ByT5 output checks preserve malformed/over-limit/protected originals explicitly;
they do not hide runtime or loading errors. Numbers/punctuation are protected.
The raw bounded repair permits up to3 words,8 edits and relative edit0.75.
These limits and all training/generation choices are assumptions in the isolated
config. Literal spelling, names, and non-English words can still be damaged.

A supplementary fixed **historical ICDAR13 replay** implements the previously
proposed Qwen insertion policies using saved outputs. It is not new Qwen inference
or transfer to the M/K Total-Text population. Its two conditions/four controls
were selected after earlier result inspection; no independent gain is claimed.

## Metrics and audit

Primary: micro per-image casefold whitespace-token multiset matching, punctuation
retained, care annotations unchanged. Duplicate predictions are capped by GT
multiplicity. Empty readings count as false-positive observations, not omissions.
Recovered/damaged counts concern capped lexical occurrences, not localized GT
identities. Report precision, recall, F1, output counts, all care denominators,
per-image predictions and changes, and conditional paired-image bootstrap
intervals (1000 draws). Intervals omit training/search variability.

Fixed-box exact counts are additionally reported for baseline/NeuSpell only,
requiring IoU>=0.5 and exact casefold one-to-one matching. The baseline must replay
524/983 on evaluation and1293/2204 on TEST. New text splits do not create reliable
new spatial boundaries, so their lexical gains are not presented as localization
or official end-to-end spotting improvements.

The selected structural/denoiser and KEEP policies are frozen before new
evaluation/test annotation parsing. Raw diagnostic arms are not automatically
promoted. Full outputs retain model histories, candidate scores, targets,
checkpoint identities, generated byte IDs/text, rejection reasons, and all
per-image outputs. Failures/OOM produce receipts and propagate.

## Execution and qualification

No packages were installed or environments created. The official pinned ByT5
snapshot alone was downloaded to the explicit external HF cache and all five
files were SHA256-recorded. Existing isolated main environment is reused.

Actual GPU: PAPER NVIDIA RTX3090/24GiB. Separate NeuSpell/ByT5 CUDA FP32 stages,
no AMP/TF32, deterministic algorithms and stage-local seeds. Record allocated
and reserved peaks, batch/input shapes, model identities, and failure state.
This does not qualify RTX3070/8GiB. Hardware choices are HARDWARE_DEVIATION.

67 focused unit tests passed, including fit-only training-label access and
injected OOM receipt/exception propagation. Real smoke1 qualified initial three stages;
smoke2 qualified the final letter-preservation controls and source-annotation
checks from an immutable1517-file source snapshot. Both completed without OOM.
Smoke results do not measure scientific efficacy.

Run from a frozen source snapshot with explicit config/output:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 /explicit/main/python -u \
  -m scripts.run_text_reconstruction_tests --mode approximate \
  --config configs/approximate/text_reconstruction_tests_v1.json \
  --output /explicit/new/output --action all
```

Add `--smoke` for24fit/4calibration/4evaluation/4TEST images and128training pairs,
one epoch. Completed output directories are immutable. New implementations use
new roots; no automatic checkpoint/model/precision fallback is available.

External root:
`/extra1/bmartelli/context-str-runs/text-reconstruction-tests-20261023-v1/`.
Read `full/status.json`, `full/results.json`, `full/comparison.md`, and the
independent audit for completion, rather than inferring it from this guide.

Scientific decision: TEXT-RECONSTRUCTION-TESTS-001 in `PAPER_AMBIGUITIES.md`.

## Completed results and interpretation

Full run SUCCESS, no runtime failure or OOM. All 1030 fit / 100 calibration / 125 evaluation /
300 TEST images completed. ByT5 executed 3 epochs / 3375 optimizer updates / 54000 training
record exposures. Epoch 1 won the frozen raw calibration criterion; epochs 2/3 had
lower training loss but more damaged calibration words. This does not support
simply adding training epochs.

Primary TEST 300 denominator 2204 care words; these are **lexical** metrics:

| Fixed raw policy | Correct | Outputs | Recovered | Damaged | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen M/K + PARSeq |1322|2824|0|0|52.59%|
| NeuSpell short insertions |1324|2824|5|3|52.67%|
| NeuSpell long insertions |1321|2824|5|6|52.55%|
| Structural classifier, threshold 0.5 |1021|2587|87|388|42.62%|
| ByT5 bounded repair |1362|2949|78|38|52.86%|
| ByT5 retaining observed letters |1365|2948|78|35|52.99%|

Fixed-box exact counts replay 1293 for baseline, 1292 for NeuSpell short and 1289 for
long: the small lexical NeuSpell increase is not a localized-recognition gain.
ByT5's lexical +43 does not establish a spatial gain. The larger output count
reduces precision 46.81%→46.30% despite recall 59.98%→61.93%.

On evaluation 125/983, baseline 536, NeuSpell short/long 538 (5 recovered / 3 damaged),
raw structural 405 (32/163), ByT5 560 (37/13), letter-preserving ByT5 562 (37/11).
These are a different population from earlier 384/983 BERT controls.

The actual bounded structural oracle reaches 589 versus 536 on evaluation and
1469 versus 1322 on TEST. This confirms some feasible text-operation potential,
not achieved selection. Structural calibration chose KEEP: no tested nontrivial
threshold met the two-damage limit. Every completion/ByT5 calibrated deployment
arm also chose KEEP. **No new method is promoted or enabled automatically.**
Consequently structural+ByT5 equals ByT5 on the original blocks in this run;
it is not evidence of successful multi-block reconstruction.

Conditional TEST bootstrap interval for letter-preserving ByT5 F1 difference:
[-0.45,+1.34] percentage points; the observed +0.40 points remains compatible with
zero. Correct-count delta interval [+19.98,+70.00] conditions on fixed fitted
models and excludes search/training variability. Forty-two images gain correct
lexical matches, 29 lose. Removing both known RGB duplicates retains the +43 net
pattern (1307→1350) and approximately +0.41 F1 points.

### What produces the apparent gain?

Post-hoc saved-output analysis, without changing the selected models/policies:

- 122 of 156 changed TEST groups only change word boundaries; 28 add a suffix and 6
 add internal/multiple letters. No prefix-only completion appears in this arm.
- 66 of 78 recovered lexical occurrences have a spelling already present in the
 bounded structural candidate pool; 12 do not. Membership does not guarantee
 simultaneous feasible selection or identify causal per-GT recoveries.
- A retrospective boundary-only ByT5 control reaches 1366, 73 recovered / 29 damaged,
 F1 53.03%. The complementary letters-only control reaches 1321, 5/6, F1 52.55%.
 These controls were defined after inspecting outcomes and are not validation
 or new calibrated deployment policies.
- Examples: `KrispyKreme→Krispy Kreme`, `TOURISTVEHICLE→TOURIST VEHICLE`,
 `KUALALUMPUR→KUALA LUMPUR`; genuine completion example `restaur→restaurant`.
 Harms include `VITABELLA→VITA BELLA`, `AHEAD→A HEAD`, `CD→COD`.
 Manual RGB spot checks of TEST 540/93 are consistent with the VITABELLA brand
 and the Krispy Kreme sign; two examples do not qualify all annotations.
- Post-hoc reading-order CER worsens 44.81%→45.55% for the letter-preserving arm.
 This diagnostic includes spaces, ordering and missing/extra text; it is not
 a pure recognizer CER or a selection metric. More matched words alone do not
 establish uniformly improved transcription.

The supported follow-up is learning **when to insert word boundaries** while
protecting complete words/names, using ByT5 as the candidate generator rather
than the current failed structural classifier. Pure missing-letter completion
has no demonstrated aggregate TEST benefit here. Multi-block fragment repair
and independent validation remain unresolved.

### Verification and artifacts

Independent audit PASS: 1555 original export hashes, 10 pretrained files,
6 calibrated policy choices, 6375 image-arm count checks, 1275 independent fixed-box
matchings and 425 feasible oracle assignments. All care denominators, empty
predictions and repeated outputs remain. 67 focused tests pass, including OOM
fault injection; no actual full-run failure/OOM occurred.

ByT5 allocated/reserved peaks 6416256512/6882852864 bytes, FP32 batch 16 with
recorded input shapes; stage 529.25 s includes loading, training and inference,
not end-to-end pipeline latency. NeuSpell peaks 822780928/876609536 bytes,
batch 32; stage 6.71 s. Models are sequential on actual RTX3090; no 8 GiB qualification.

Under the external root, preserve:

- `full/results.json` and `full/comparison.md`: all raw and calibrated policies.
- `full/independent_audit.json`: independent identity/count/constraint checks.
- `full/analysis_final.json`: post-hoc types, subsets, CER, sensitivity and examples.
- `full/selection_frozen_before_evaluation.json`, `denoiser_history.json`,
 `denoiser_training_pairs.json`, model/checkpoint hashes and per-image outputs.
- `qwen_replay/results.json`: historical-only four-control replay. It reproduces
 506/849 (5/1) with Florence and 505/849 (3/0) with InstructBLIP under unbounded
 letter preservation. The old short limits give 502 (1/1) and 503 (1/0), respectively.

Full primary result SHA256:
`3f1a5a477a04b0986801a52134cf8b5931e0ebb86e7632137a0d26d17d873fe4`.
Selected fine-tuned checkpoint SHA256:
`d9bc8caebc5ecc16a5af985080f739e28f93377f9f02bbaeab6316dfabf88327`.
