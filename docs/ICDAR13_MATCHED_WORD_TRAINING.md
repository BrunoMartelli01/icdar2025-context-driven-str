# ICDAR2013 symbol-free word F1: audit and matched Total-Text loss controls

**Approximate reproduction / research extension.** The requested objective is
U-Net localization followed by genuine PARSeq, measured by word F1 without
punctuation. This study does not use a router or run DeepSolo. Every reported
population therefore has DeepSolo0/N images (0.00%). It does not establish
official polygon Hmean, Table5 recognition accuracy, or strict reproduction.

Before implementation, inspect the local paper sections3.2,3.3 and4.1, the
existing native refit, and the pinned Total-Text source training contract.
Decisions are recorded in `ICDAR13-MATCHED-LOSS-001` in
`docs/PAPER_AMBIGUITIES.md`.

## What the historical comparison actually establishes

The completed native refit already uses filled full-word targets, rather than
glyph-only masks. All229 TRAIN images /849 care words passed the native384
topology and official Task1 word-envelope checks. Original source annotations
and masks remain intact. Repeating conversion alone would not resolve its
observed generalization gap: FIT183 F1 95.3437%, selection23 88.7097%, CAL23
72.0430%, TEST233 71.8511%, all with DeepSolo0/N (0.00%). See
`docs/ICDAR13_WORD_REFIT.md` and its preserved completion/recheck evidence.

The previous corrected Total-Text pipeline gives TEST F1 73.3237%, with
DeepSolo0/233 (0.00%). It includes candidate selection, a graph, an RGB-backed
verifier and a text editor. The native refit uses direct text/kernel proposals
and PARSeq with a sequence-confidence cutoff. This is a pipeline comparison;
it does not isolate the effect of the training dataset or word masks.

| Recipe feature | Qualified Total-Text source | Completed native refit | New matched pilot |
| --- | --- | --- | --- |
| Filled separated words / ownership | Yes | Yes, all849 words | Reuse exact same targets |
| Letterbox / invalid padded canvas | Yes | Yes | Yes |
| Residual attention skip floor | .25 | .25 | .25 |
| Word kernel / distance fraction | Yes / .5 | Yes / .5 | Yes / .5 |
| Separator penalty | 3 | 3 | 3 |
| Source text/kernel loss | Masked per-image BCE+Dice | Text BCE + balanced kernel BCE | Compare both losses, otherwise identical |
| Initialization | Qualified Total-Text continuation | Every source tensor, strict load | Identical complete source state |
| Encoder / BN | Frozen | Frozen and partial layer3/4 controls; BN frozen | Frozen, with encoder BN frozen |
| RGB / native maps | 512 /256 | 768 /384 | 768 /384 after failed512 topology preflight |
| Adam decoder LR | Constant1e-5 | Cosine1e-5 to1e-6 | Constant1e-5 |
| Physical batch / accumulation | 8 /1 | 4 /2 | 8 /1 |
| New training budget | Source continuation50 epochs on1130 FIT | 100 epochs per arm on183 FIT | 30 epochs per loss arm on183 FIT |
| Model selection | Word-proposal geometry | PARSeq lexical F1 | PARSeq lexical F1, source epoch0 eligible |
| Additional graph/verifier/editor | Added by the corrected comparison pipeline | Absent | Absent |

Other historical Total-Text interventions (photometric, Fourier, Haar and
partial encoder) are separately documented ablations. Their presence in the
repository does not establish that they all improved OCR or should be combined.
The selected source here is the authenticated Text+Kernel control used by the
preceding native refit. This pilot restores its loss in one arm and keeps the
other loss as a matched control; only loss differs between the two new arms.
Differences from the historical100-epoch refit remain explicit confounders.

## Failed512 preflight and retained word supervision

The initial512/native256 preparation stopped before GPU training. Auditing the
actual loader on all229 images found owner20 of `icdar13:train:126` absent after
resize/separator processing:848/849 words remain. All other owner maps pass.
The failed source snapshot and detailed audit remain under
`/extra1/bmartelli/context-str-runs/icdar13-matched-word-training-20261028-v1/`.
Do not drop that word, reinterpret the denominator, or relabel the failed run.

The qualified revision retains768/native384 and checks all849 words again.
Masks, validity, separators and owners are read from the exact previous target
manifest, authenticated by SHA256. No raw GT is overwritten and no TEST owner
map is created.

## Frozen experimental boundaries

Keep the existing RGB-disjoint183 FIT /23 selection /23 CAL /233 TEST
partitions. Only FIT images enter gradients. Both30-epoch branches start from
the same strictly loaded Total-Text state, fresh Adam1e-5, weight decay.01,
physical8, seed20261028 and identical flip/rotation sample order. No AMP,
quantization, layer replacement, alternate recognizer or missing-key relaxation.

Evaluate source epoch0, every5 epochs and the final epoch on selection23 under
text.65/kernel.5, original-pixel padding4, maximum50 crops and native sequence
probability.9. Keep the earliest exact F1 tie. Compare selected checkpoints and
unchanged source using the predeclared finite grid: text [.5,.65,.8], padding
[0,2,4], sequence cut [0,.4,.7,.9]. Include the source in final selection and
freeze the exact winner before any CAL/TEST scoring. Neither a CAL result nor a
TEST result can trigger new thresholds, weights or training in this experiment.

The primary metric counts per-image multisets after Unicode casefold and
`isalnum` filtering, then aggregates TP/FP/FN across images. No localization
IoU is required. Preserve the inherited native256 ignore policy and all care
GT; empty predictions retain FN. Explicitly emit no word for empty or
alphanumeric-empty decodes before evaluation, retaining all raw crop decodes.
This small emission extension is recorded separately from raw-output results.

Runtime is the existing main environment on the observed RTX3090, FP32,
RGB768/native384, one segmentation model with pinned PARSeq. Save allocated
and reserved peaks, exact checkpoint/config/source hashes, masks, crop hashes,
raw/emitted readings and per-image counts. OOM and inference errors propagate.
Hardware and precision qualify this run only; no8GB or latency claim follows.

Implementation: `segmentation/icdar13_matched_training.py` and
`scripts/run_icdar13_matched_word_training.py`.
Profiles: `configs/approximate/icdar13_matched_word_training_v1.json` and
`configs/approximate/icdar13_matched_word_training_local_paths_v1.json`.
Tests: `tests/segmentation/test_icdar13_matched_training.py`.

Independent count/selection replay:
`scripts/audit_icdar13_matched_word_training.py`. Audited tables, learning curves
and a portable selected-expert manifest:
`scripts/report_icdar13_matched_word_training.py`.

For common-emission historical comparisons, the independent auditor also reads
the original233 official Task1 CSV files directly and authenticates every cached
sample. It rejects only emitted readings with no alphanumeric character. The
native refit's14 such readings account for a change from71.8511% to72.3343%
without adding a correct word (753TP). The corrected Total-Text pipeline's8
such readings account for73.3237% to73.6077% (760TP). These controls are saved
separately and must not be called a training improvement. Both have
DeepSolo0/233 (0.00%).

## Qualification and execution

Nineteen focused source-loss, gradient, ownership, encoder, emission and split
tests passed. A genuine full-batch8 CUDA backward/optimizer smoke passed both
losses, then completed selection, separate CAL and TEST control inference and
emitted-output count replay on four images per evaluation population. These
are smoke-test results, not a quality estimate. Both backward arms peaked at
8,736,153,600 allocated bytes; reserved peaks were12,071,206,912 and
12,012,486,656 bytes. No OOM occurred. Later additive guards record OCR peaks
and verify the selected checkpoint's fixed-policy replay in the full run.

External run: `/extra1/bmartelli/context-str-runs/icdar13-matched-word-training-20261028-v2/`.
The `smoke/` and `full/` trees preserve independent resolved configurations and
source snapshots. Both full30-epoch arms and the final CAL/TEST evaluation are
complete. All183 fitting images occur in each of60 epochs (10980 image passes),
with identical per-epoch sample-order hashes across the two losses.

## Completed results and interpretation

At the fixed checkpoint policy, the BCE control selects epoch20 with selection
F1 87.3016%; BCE+Dice selects epoch30 with89.6000%. The finite selection-only
grid then chooses BCE+Dice epoch30, text.5/kernel.5, padding4 original pixels,
maximum50 crops and PARSeq native sequence probability.9. Selection F1 is
90.3226% (56TP,4FP,8FN), with DeepSolo0/23 (0.00%). The checkpoint SHA256 is
`0e9a4bcc51ca1901e6b1c119c09f98c326703946e6ed92c0cba8fdcbd4265f65`.

All233 official TEST images and1095 care words are retained, with zero inference
failures. Every row below has the same explicit no-alphanumeric emission rule.

| Expert | TEST TP | FP | FN | Lexical micro F1 % | DeepSolo images/total (%) |
| --- | ---: | ---: | ---: | ---: | --- |
| Historical native word refit, common emission | 753 | 234 | 342 | 72.3343 | 0/233 (0.00%) |
| Unchanged Total-Text source at matched768/fixed policy | 769 | 257 | 326 | 72.5130 | 0/233 (0.00%) |
| Historical corrected Total-Text pipeline, common emission | 760 | 210 | 335 | 73.6077 | 0/233 (0.00%) |
| Selected new BCE+Dice pilot | 772 | 223 | 323 | **73.8756** | 0/233 (0.00%) |

The selected pilot improves the native refit by1.5413 percentage points under
common emission, with19 more correct words and11 fewer false positives. The
observed margin over corrected Total-Text is only0.2678 points; do not claim a
robust general advantage from this small difference. The original native
71.8511% to new73.8756% change also includes0.4831 points from rejecting its14
symbol-only predictions, which the common-emission comparison isolates.

Separate CAL23 remains weak: the selected expert gives72.7273% (68TP,24FP,27FN),
versus72.2513% (69TP,27FP,26FN) for the fixed unchanged source; both have
DeepSolo0/23 (0.00%). The90.3226% selection score is optimistic relative to CAL
and TEST. The new result improves the requested observed F1, while small,
reused validation and limited native data remain generalization limitations.

The matched new loss arms isolate loss on selection, where BCE+Dice performs
better. The new-versus-historical TEST gain combines loss, optimizer/batch
recipe, training duration and validation-selected crop policy; it cannot all be
attributed to Dice. No full TEST comparison of both new loss arms was selected
after observing results, and no TEST threshold was promoted.

Both full training arms run FP32 RGB768/native384, physical8 with no accumulation
on RTX3090. Allocated/reserved training peaks are8,998,063,104/12,079,595,520
bytes (BCE) and8,996,948,992/12,060,721,152 (BCE+Dice). No OOM occurred. This is
not an8GB qualification. Main inference and targets remain in the original
qualified environment. The plot uses an existing separate environment solely
to render saved numerical values, with no model or target computation there.

Independent audit PASS authenticates740 frozen sources, all60 epochs, checkpoint
and policy selection, and every TEST image's lexical counts from direct official
Task1 CSV parsing and an independent Counter implementation. The full audit is
`full/independent_lexical_audit.json`; results are `full/results.json`, per-sample
outputs are under `full/evaluation/`, and `full/report/` contains a numerical CSV,
learning-curve/comparison PNG, selection-geometry diagnostic, summary and the
fully identified `selected_expert.json` manifest.

Preserved failures: the512 topology preflight, the first report's missing
Matplotlib dependency, and a report-only interpreter-path error that followed a
virtualenv symlink into the bare Python executable. The latter was corrected by
preserving the explicit virtualenv path. Failed report directories remain under
the full run. No package was installed, no scientific prediction was replaced,
and no failure was hidden or counted as a successful experiment.
