# Recognition and Context Foundation Decisions

This document defines only the runnable Macro-Milestone 5 `APPROXIMATE` path.
It does not resolve or weaken any `STRICT` blocker in `PAPER_AMBIGUITIES.md`.

| Component | Classification | Approximate decision | Scientific boundary |
|---|---|---|---|
| PARSeq artifact/runtime | `ASSUMPTION` | Use `parseq-bb5792a6.pt` at SHA256 `bb5792a...cd00` with official runtime `1902db043c...`; strict state-dict load, official 32x128 RGB transform, AR decoding and one refinement pass. | The paper does not bind this pair to its PARSeq-S results. |
| PARSeq confidence | `ASSUMPTION` | Preserve every decoded-character probability and expose their arithmetic mean as convenience sequence confidence. | The paper gives no confidence aggregation; scoring does not consume this mean. |
| PARSeq proposal-ranking scores | `ASSUMPTION`, diagnostic only | Compare three independent, untrained rankings: the upstream evaluation product of decoded greedy-path probabilities (including EOS), mean character log-probability, and minimum character probability. | These scores are not combined, do not replace the existing pipeline confidence, and are evaluated on TRAIN only after proposal generation/OCR are frozen. |
| Full-image T1 | `UNRESOLVED` | Permit the adapter call, label its role, and retain the unresolved warning in metadata. | A single-instance 32x128 recognizer applied to a full scene is not scientifically defined. |
| Multiple crops to T3 | `UNRESOLVED` | Preserve the rank-1/largest-component `ASSUMPTION` branch and add a separate `REPOSITORY_SPECIFIED` branch that reproduces author-runner commit `2e8b544`: score every crop and select the first maximum paired `C3`. | Neither branch is paper-established; they use separate resolved configs and output roots, and `L-004` remains blocking. |
| BLIP-2 candidate | `REPOSITORY_SPECIFIED` | Pin `Salesforce/blip2-flan-t5-xl@0eb0d3b...`; use the repository prompt and generation values. | This is EVA-CLIP-g-family/Flan-T5-XL, not the paper's unresolved “ViT-B” model. |
| BLIP-2 decoding | `ASSUMPTION` | Make greedy/beam behavior explicit with `do_sample=false`; treat 40/80 as new tokens. | Prompt completeness, length unit, and paper decoding remain unresolved. |
| BLIP-2 placement | `HARDWARE_DEVIATION` | Physical-host FP16 `single_device` placement OOMed during `model.to(cuda:0)` at 7,171,370,496 allocated / 7,656,701,952 reserved bytes. The sole M5 adaptation under qualification is FP16, `quantization=none`, `placement=cpu_offload` using Transformers `device_map="auto"` and Accelerate module dispatch. | This remains `APPROXIMATE`; no quantized or alternate-model branch is enabled. The profiler records the resolved module map, load/generation/total time, process peak RSS, and GPU peaks. |
| MPNet | `REPOSITORY_SPECIFIED` | Pin `all-mpnet-base-v2@e8c3b32...`, mean pooling, normalized embeddings, truncation at the artifact's 384-token limit. | The paper does not identify this checkpoint or embedding protocol. |
| MPNet precision | `ASSUMPTION` | Float32 in the approximate configuration. | Precision is absent from the paper. |
| Lexical nominal path | `REPOSITORY_SPECIFIED` | RapidFuzz 3.11.0 `fuzz.ratio(raw, raw)/100`. | Other named preprocessing modes are diagnostics only; partial/token ratios are not nominal. |
| Score interpretation | `UNRESOLVED` | Keep `figure2_independent_max`, `selected_candidate_paired`, and `repository_paired_threshold` separate. The repository branch reproduces scorer commit `87b90db` and runner commit `2e8b544`: threshold `max(C1,C3)` and independently return the semantic winner. | Repository provenance upgrades that branch from an assumption to `REPOSITORY_SPECIFIED`, but no strategy is promoted to strict. |
| Ties/empty values | `ASSUMPTION` | Exact ties choose T1; missing strings become empty strings and remain visible. Fallback is needed only for `C < 0.8`, not equality. | Paper edge-case behavior is unknown. |
| Caption cache | `ASSUMPTION` | Key by canonical decoded-pixel SHA256 (shape/dtype/bytes), model revision, prompt SHA256, and canonical generation-config SHA256; caller supplies an external cache root. | Cache files are generated artifacts and remain outside git; hashing source-file bytes instead could distinguish lossless-equivalent images. |
| Machine output | `ASSUMPTION` | Schema version 1 retains masks/blocks, T1/T2/T3 metadata, all four primitive scores, all three decisions, and per-strategy `needs_fallback`. | M5 never invokes or substitutes DeepSolo. |
| Trace diagnostics | `ASSUMPTION` | Report exact raw-string membership and one-to-one multiset overlap both case-sensitively and with Unicode `casefold`, after verifying index/result/intermediate hashes. Keep the interpretations side by side. | These observations define neither the paper's matching nor its denominator. `accuracy` remains null, failed samples suppress fractions, and every Table 5/6/7 claim remains false. |
| PARSeq/DeepSolo alignment diagnostic | `ASSUMPTION` | Retain every crop/detection pair; compare the exact padded source-coordinate PARSeq crop box with the axis-aligned envelope of each DeepSolo polygon; report both asymmetric coverages, IoU, center inclusion, raw/casefold exactness, and RapidFuzz ratio. | Polygon-envelope comparison and diagnostic thresholds are not paper evaluation rules. T1 is a full-image input and cannot be localized to one detection. Accuracy remains null and Table 5/6/7 claims remain false. |
| PARSeq crop/GT parameter sensitivity | `ASSUMPTION` plus CPU `HARDWARE_DEVIATION` | Keep paper-specified localization mechanics fixed; sweep `Amin={50,150,300}`, padding `{0,6,12,24}`, and connectivity `{4,8}` over frozen masks. Associate crop/GT pairs by geometry before measuring exact and RapidFuzz text. Require the `150/6/8` crop boxes, pixels, and texts to reproduce the frozen CUDA trace. | The grid, GT inclusive-to-exclusive conversion, and geometry-first matching are diagnostics, not parameter-selection or evaluation protocols. New variant crops were not independently cross-device replicated. Accuracy remains null. |

The literal whole-string feasibility diagnostic is investigative only. With
`alpha=0.6`, `beta=0.4`, and `tau=0.8`, even perfect semantic similarity needs
lexical similarity of at least 0.5. Representative short STR strings versus a
long caption scored 0.0000–0.0197, yielding optimistic confidence 0.6000–0.6079.
This supports warning `S-003`; it does not justify changing the nominal matcher.

The physical repository-aligned IDs 1--10 comparison confirms that crop aggregation is
not the cause of the observed routing failure. Scoring every crop changed selected T3 on
7/10 samples and `C` on 5/10, but the largest `C` was only
`0.26995353443669656`; fallback remained 10/10 and every final output was unchanged.
Since `L <= 1`, the configured formula requires `S >= 2/3` to reach `tau=0.8`; the
largest semantic score across T1 and every crop was `0.39617995924870714`, making all ten
routing decisions invariant to any alternative lexical matcher bounded by one.
This qualifies the public repository control flow as an APPROXIMATE diagnostic while
leaving the paper's reported 73% no-fallback behavior unresolved.

The hash-linked first-ten alignment report shows a different upstream mismatch. Although
66/76 PARSeq crops overlap a DeepSolo envelope and positive crops have median best coverage
`0.9864153811332046`, only 31/54 DeepSolo detections overlap any crop, 23/54 detection
centers lie inside any crop, and just 6/54 detections have at least 50% envelope coverage.
Only 2/31 spatially best positive pairs are case-fold exact and 5/31 reach case-folded
RapidFuzz ratio `0.8`. Thus the recorded recognizers are not generally observing the same
word instances: PARSeq often receives a fragment of a larger DeepSolo word envelope, and
many DeepSolo words have no retained crop. The envelope method remains an `ASSUMPTION`;
the next defensible diagnostic is to use frozen ICDAR13 ground-truth word boxes as an
independent spatial bridge, without promoting that diagnostic to recognition accuracy.

The 24-point crop/GT grid reproduced all 76 baseline CUDA crop texts exactly on CPU.
At `Amin=150`, padding 6, connectivity 8, only 41/66 GT words overlap a crop and only
7/66 reach best IoU `0.5`; geometry-first text is case-fold exact for 5/66. Changing
`Amin` to 50 or 300 changes crop cardinality but not those outcomes. Connectivity 4
weakens spatial coverage. Padding 24 increases permissive overlaps and fuzzy matches,
but does not increase the number of IoU-0.5 pairs and reduces exact recognition within
that strong-geometry subset. Therefore the first-ten result supports neither changing
the frozen repository values nor tuning them for a reported score; it instead points to
the upstream mask/component geometry as the dominant unresolved issue.
