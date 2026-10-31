# Four occlusion-inspired component experiments

2026-10-22. **Approximate reproduction / exploratory research.** User-authorized
implementation and execution of four tests. This is not an exact implementation
of Das et al., official polygon spotting, paper Table 5, independent validation,
or an evaluation of the complete BLIP-2/MPNet/DeepSolo routing pipeline.

## Implementation and evidence

- `localize/occlusion_four_tests.py`: GT-blind letterbox instance alternatives,
  RGB MSER proposals/grouping, pre-OCR features, reading agreement and KEEP policy.
- `scripts/run_occlusion_four_tests.py`: source/artifact checks, real neural
  execution, TRAIN fitting/calibration, frozen evaluation, cold budget execution
  and independent SciPy matching audit.
- `configs/approximate/occlusion_four_tests_v1.json`: all scientific choices,
  explicit local artifact paths, model/checkpoint hashes and parameters.
- `tests/localize/test_occlusion_four_tests.py`: interfaces, geometry, conflicting
  groups, budgets, confidence/context, namespace isolation and failure regressions.

Before implementation, the original paper sections 3.2–3.5 and Algorithm 1,
Das et al. section 3.1/equation 1 and public notebook, existing mask-crop selector,
M/K/Gap decoder, qualified PARSeq adapter and BERT scoring code were inspected.

## Frozen populations and protocol

Use the earlier **1030-fit Gap** epoch 1, not the later full1255 refit. Source M/K
and Gap are strictly loaded and frozen. Existing IDs define 1030 selector-fit,
100 selector-calibration, 125 TRAIN-evaluation and all300 TEST images. The125
evaluation images did not enter Gap updates, but prior checkpoint/parameter
selection and repeated historical exploration prevent independent validation
claims. The M/K source saw the100 calibration images during its own training;
they are held out only from new selector fitting.

Every generator and all OCR/context inference finish before any new classifier
uses annotations. Classifiers fit only1030, policies calibrate only100, and the
selection is saved and hashed before loading125-evaluation or TEST transcripts.
Identity/KEEP is available; correction calibration ties prefer fewer edits.

All arms share micro maximum one-to-one original unpadded bbox IoU>=0.5 and
exact Unicode casefold, preserving punctuation. Every care word and output,
including duplicates/empty readings, counts. Ignored GT is excluded, but no
prediction is neutralized with an annotation mask. This latter choice differs
from the earlier Gap report's wholly-invalid-instance exclusion, so its old
1293/1299 counts are not mandatory replay targets here.

Known exact RGB duplicates TEST578=TRAIN212 and TEST656=TRAIN81 stay in primary
TEST300. A separate sensitivity excludes exactly those two. Unknown checksum
overlap fails preflight. Neither historical TEST300 nor its298-image subset is
an untouched benchmark. OSTD and dedicated natural-occlusion labels are not used;
the suite cannot establish occlusion-specific restoration performance.

## Four comparisons

1. **Selector transfer:** 2x2 M/K versus previous1030fit Gap, each with original
   assembly versus selected split/merge alternatives. Preserve all original
   instances, add a bounded pool and resolve conflicting replacements. A shared
   small MLP fits actual localized-and-exactly-read correctness. The experiment
   changes postprocessing, not segmentation or recognition weights.
2. **RGB candidate generation:** standard-reference MSER versus an explicitly
   **SS-MSER-inspired** cross-delta rectangle-stability filter, with shared
   geometric/color grouping and learned admission. Test standalone proposals and
   their augmentation of the original M/K outputs. RGB features/admission do not
   use U-Net probabilities, so standalone inference does not require U-Net.
   The weighted soft-set equation is underspecified and differs from the public
   notebook; the adaptation is an ASSUMPTION, not exact Das reproduction. Full
   pool coverage is a diagnostic ceiling, not the selector's achieved accuracy.
3. **Multi-view reading correction:** original RGB box plus10%/25% of box-height
   padding, clamped/deduplicated. Same PARSeq AR/refine1 produces competing strings;
   confidence/agreement gates and optional pinned BERT local-OCR context choose
   among them. Original detection geometry remains fixed. BERT sees original OCR
   neighbors, never GT or recursively corrected text. No new caption, CRNN,
   lexicon-derived candidate, or beam-search experiment is implied.
4. **Pre-OCR budgets:** a second classifier receives RGB/geometry/provenance/map
   features only. Compare learned and area-based preselection at10/20/40 unique
   RGB crops per image, with the same frozen downstream replacement policy.
   Unselected OCR scores cannot influence that policy. Full-pool OCR supports an
   offline replay; a separate empty-cache execution actually recognizes every
   selected TEST crop and checks its text against replay. Per-policy receipts
   report actual stage time/VRAM/read counts. Time includes PARSeq loading and
   excludes segmentation/proposal generation/feature extraction/preselection;
   it is not end-to-end latency or a detector speed comparison.

## Artifacts and execution

External root:
`/extra1/bmartelli/context-str-runs/occlusion-four-tests-20261022-v1/`.
The full experiment runs from `source_snapshot/` with1105 source/config/test
files hashed in `source_snapshot_manifest.json`; weights/datasets remain external.

From that snapshot, using the existing main interpreter and a fresh output:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 /explicit/main/python -u \
  -m scripts.run_occlusion_four_tests --mode approximate \
  --config configs/approximate/occlusion_four_tests_v1.json \
  --output /explicit/new/output
```

Add `--smoke` for8fit/4calibration/4evaluation/4TEST images. A cached resume refuses
changed configuration, source code or sample identities. Inputs and per-sample
numeric maps are checksum-verified; source weights/manifests are rechecked at end.
Outputs retain raw map arrays, proposal boxes, pixel hashes, readings, confidence,
local-context scores, classifier checkpoints/history, calibration grids, policies,
per-sample outputs/matches, aggregate JSON and Markdown. No individual crops are
saved. Errors and OOM propagate with a failure receipt.

## Qualification and limitations

Actual host: PAPER, NVIDIA RTX3090/24GiB. Explicit CUDA FP32, no AMP/TF32, seeded
deterministic algorithms; batch1 segmentation, batch32 PARSeq, sequential BERT.
CPU MLP fitting uses the existing environment. No installation/download.

61 focused tests pass. Initial test invocation used a nonexistent test path and
ran no tests; corrected paths pass. Real smoke1 failed at the Torch-to-NumPy
canvas boundary; its failure handler also had a filename-concatenation bug.
Both were corrected, and that failed protocol is preserved. Smoke2 completed
real segmentation/PARSeq/BERT but exposed baseline copies lacking their recognized
pool crop hashes after JSON serialization. Explicit rebinding and a regression
test fixed the boundary. Smoke3 completed all four comparisons and independent
matching; final smoke repeats the final explicit-threshold implementation.
BERT reports a dependency deprecation warning and the four explicitly allowed
unused pretraining-head/pooler keys; missing/mismatched/unapproved keys fail.

Classifications and remaining scientific questions are recorded in
`docs/PAPER_AMBIGUITIES.md`, OCCLUSION-FOUR-TESTS-001. Completion/results must be
read from the full run's `status.json`, `results.json`, `independent_audit.json`
and `budget_execution.json`, not inferred from implementation or smoke success.

## Completed results (2026-10-22)

All1030 fit,100 calibration,125 evaluation and300 TEST images completed. TEST denominator2204care words; internal evaluation983. The immutable source snapshot and all pinned inputs remain valid. No full-run failure or OOM.

| TEST arm | Correct words | Outputs | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| mk_baseline | 1293 | 2824 | 45.79% | 58.67% | 51.43% |
| mk_selected | 1307 | 2755 | 47.44% | 59.30% | 52.71% |
| gap_baseline | 1299 | 2848 | 45.61% | 58.94% | 51.43% |
| gap_selected | 1312 | 2774 | 47.30% | 59.53% | 52.71% |
| mser | 70 | 343 | 20.41% | 3.18% | 5.50% |
| ss_mser_inspired | 56 | 295 | 18.98% | 2.54% | 4.48% |
| mser_augment_mk | 1315 | 3112 | 42.26% | 59.66% | 49.47% |
| ss_mser_inspired_augment_mk | 1311 | 3073 | 42.66% | 59.48% | 49.69% |
| correction_visual | 1309 | 2824 | 46.35% | 59.39% | 52.07% |
| correction_context_0.05 | 1300 | 2824 | 46.03% | 58.98% | 51.71% |
| correction_context_0.1 | 1296 | 2824 | 45.89% | 58.80% | 51.55% |

Selector transfer is positive for both M/K and Gap:19recovered/5damaged and18/5, respectively. M/K delta+1.28F1points, paired conditional95% interval[+0.79,+1.80]; Gap delta+1.29[+0.80,+1.81]. Evaluation125 also improves524→528 and526→530. These intervals condition on the trained models/policies; they exclude training variation and historical search. Gap alone adds6words but its F1 delta is-0.007points, interval[-0.31,+0.38], so extra decoder cost has no demonstrated F1 benefit here.

RGB MSER does not replace U-Net in this implementation. Standalone MSER/SS-inspired read70/56words. Their complete bounded pools contain only102/86 potentially correct matches, a GT-assisted diagnostic rather than achieved selection. Augmentation recovers words but worsens F1 by1.96/1.75points through extra false positives. This rejects the tested grouping/filter/cap configuration, not every possible MSER method or the original trained Das system.

Visual crop correction recovers32/damages16, net+16; F1 delta+0.64points, conditional95%[+0.08,+1.18]. Local-OCR BERT controls read1300/1296 versus1309 visually: no added observed benefit. No occlusion-specific result, scene-caption ablation or claim about all language models follows.

### Recognition budget tradeoff

| M/K control | Actual unique OCR crops | Correct words | F1 |
| --- | ---: | ---: | ---: |
| mk_selected | 4177 | 1307 | 52.71% |
| budget_mk_learned_10 | 2399 | 1104 | 52.90% |
| budget_mk_learned_20 | 3418 | 1255 | 53.76% |
| budget_mk_learned_40 | 3990 | 1303 | 53.22% |

At the same20-crop maximum, learned and area-based ranking each request3418unique crops: learned reads1255words versus1140, F1+2.18points, conditional95%[+0.71,+3.76]. At10crops the gain is246words/+6.79points. This is evidence for learned ranking under a budget, not free accuracy.

Relative to the complete M/K selector,20-crop preselection saves18.2% of recognition requests but loses52correct words;40-crop saves4.5% and loses4. Precision/F1 and recall therefore trade off. The20/40 controls are descriptive fixed-budget results, not a TEST-selected deployment winner. Cold replay actually inferred every requested crop and reproduced all reading strings. Stage times include loading and exclude detection, features and preselection; they do not qualify total latency.

### Audit and interpretation

Primary independent matcher:9775image-arm checks PASS. Additional independent scalar-IoU/matching audit:6900TEST image-arm checks;31390original-RGB crop identities and3600per-image budget checks PASS. The full-pool cold reference also passed. Seed20261022,10000paired image bootstrap draws; script and analysis hashes are saved. The two-known-duplicate sensitivity retains the main patterns (M/K1278→1292, Gap1284→1297).

Measured peak across main stages:699942912allocated/960495616reserved bytes on the actual RTX3090; noOOM. This is not RTX3070/8GB qualification. All26available source-component hashes match their original M/K contract. No packages were installed and no scientific pipeline defaults or model weights were changed.

Priority supported by this run: discrete selector transfer, then pre-OCR ranking at an explicitly chosen recall/cost target. Visual correction is a smaller exploratory gain requiring stricter damage control. Do not promote the tested MSER replacement or local-BERT correction. No global ranking against prior experiments with different ignore/normalization/budgets is inferred.

Full result SHA256: `34b3f1394cfa4317f6a78a46f63f2ed2482ba873066af63b622991ca76c74aa5`.

Full report: `/extra1/bmartelli/context-str-runs/occlusion-four-tests-20261022-v1/full/comparison.md`; paired analysis: `/extra1/bmartelli/context-str-runs/occlusion-four-tests-20261022-v1/full/analysis.md`; structured audit: `/extra1/bmartelli/context-str-runs/occlusion-four-tests-20261022-v1/full/analysis.json`.

Reproduce paired analysis using `/extra1/bmartelli/context-str-runs/occlusion-four-tests-20261022-v1/analyze.py` with the frozen snapshot on PYTHONPATH; cold full-pool reference is `/extra1/bmartelli/context-str-runs/occlusion-four-tests-20261022-v1/full/full_budget_reference`.
