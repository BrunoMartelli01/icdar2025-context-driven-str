# Improving scene-caption quality

Approximate reproduction / caption-quality extension. The user requests captions
similar in quality to the paper's Figure 4 and explicitly permits longer, more
detailed descriptions. This follows the frozen three-dataset comparison in
[PAPER_CAPTION_ROUTER_COMPARISON.md](PAPER_CAPTION_ROUTER_COMPARISON.md).
Scientific decisions are recorded as `CAPTION-DETAIL-001` in
[PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md).

## Quality target and selection boundary

The target is a coherent paragraph describing visible text-bearing objects,
colors, shapes, layout, spatial relationships and surroundings. More words do
not count as improvement when they introduce unsupported objects, transcriptions,
locations, purposes or stories. A tight crop may legitimately have a shorter
caption. No hand-authored scene facts, annotations, PARSeq readings, rewritten
sentences or repetitive length filler are appended to model outputs.

Nine development TRAIN images are fixed by frozen RGB hashes: three each from
Total-Text, ICDAR2013 and ICDAR2015. Their visible facts were inspected before
reading candidate outputs. Choose a profile on these images by factual grounding,
useful concrete detail, coherence and repetition. Recognition F1 and TEST labels
do not enter this selection. Nine separate CAL images qualify the frozen choice;
the exact Figure 4 scene (`total_text:1194`) is a later diagnostic. These small
samples and earlier image exposure do not establish an untouched population
factuality benchmark.

The original captioner, all earlier results, source snapshots and raw candidate
outputs are preserved. Improved non-BLIP-2 captioners are explicit model ablations,
never silently promoted into STRICT mode or labelled as the author's BLIP-2.

## Predeclared candidates and observed limitations

The first six profiles compare BLIP-2 and native InstructBLIP with a larger
token ceiling, anti-repetition generation, single-paragraph prompts and raw
multi-aspect answer assembly. BLIP-2 remains short and often repeats the same
observation; InstructBLIP produces longer paragraphs but repeatedly introduces
unseen vehicles, people, objects, locations and explanatory speculation.

Two simpler InstructBLIP prompts do not remove those factual errors. Florence-2
native detailed-caption tasks are more grounded on many examples but often give
only short generic descriptions, with too little visual detail for the target.
Native batched Florence decoding initially exposed trailing PAD tokens; standard
tokenizer special-token removal fixes this decoding artifact, with raw generated
tokens retained. No authored caption text is substituted. None of those initial
profiles is promoted as matching the illustrated paper quality.

The next candidate is the separate native `Qwen/Qwen2-VL-7B-Instruct` model at
revision `eed13092ef92e448dd6875b2a00151bd3f7db0ac`. The official model card and
Transformers integration were inspected; the installed Transformers 4.47.1
already provides its native class, so no package installation or upgrade is used.
Its documented image-processor controls are used explicitly:
[official model card](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/blob/eed13092ef92e448dd6875b2a00151bd3f7db0ac/README.md),
[native Transformers documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen2_vl.md).

Two image-only prompts request concrete visual descriptions without invented
named locations or stories. Native BF16/SDPA inference uses one image, greedy
generation and an explicit 256–1024 visual-token image budget. Generated-token
suffixes, chat prompts, actual image grids, original shapes, raw captions and
allocated/reserved GPU peaks are saved. Model-loading incompatibilities, OOMs
and empty outputs propagate rather than triggering an alternate model.

All five official weight-shard SHA256 values and the exact revision are recorded
before download. The writable disks have less free space than the 16.6 GB model;
the isolated acquisition therefore uses a temporary RAM-backed cache after
checking available host memory. Raw scientific results and identities remain on
persistent storage. The cache is not a durable checkpoint archive: independently
rerunning the model requires restoring the same hash-verified bytes. No existing
datasets, weights or user files are removed to make room.

## Implementation and evidence

Core code: `caption/detailed_scene.py`, `caption/florence_scene.py`,
`caption/qwen_scene.py`, `scripts/probe_detailed_scene_captions.py`.
Profiles: `configs/approximate/detailed_scene_captions_v1.json`,
`configs/approximate/detailed_scene_captions_v2.json`,
`configs/approximate/detailed_scene_captions_qwen_v3.json`.
Focused tests: `tests/caption/test_detailed_scene.py`,
`tests/caption/test_qwen_scene.py`; the existing native batch and caption
foundation tests also apply.

External root: `/extra1/bmartelli/context-str-runs/caption-detail-20261030-v1`.
`sample_protocol.json`, `development_visual_facts.json`, every resolved probe
plan and every native image/profile output remain available there. The original
six-profile and next four-profile sources have separate immutable snapshots.

Final selection, real qualification and illustrative output are recorded after
the native candidate run completes. Caption-quality findings must not be
presented as new recognition F1 without a separate matched router evaluation.

## Completed qualification and recognition ablation

The frozen `qwen_scene_detail` prompt is qualified on all nine CAL images with
the exact pinned model, both as single-device batch1 and native left-padded
batch2. Native token/RGB/prompt/checkpoint audits pass. Batch2 has 79–126 words
per CAL caption (mean99.56), and measured RTX3090 peak allocated/reserved memory
is 21,261,298,176/23,657,971,712 bytes. Only3/9 captions exactly match batch1;
batch arithmetic can change output, so the final experiment explicitly uses
batch2 rather than claiming scalar equivalence. `qualification_batch2_receipt.json`
in the external root binds every CAL output and records the direct visual review.

Visible objects and spatial relationships are substantially better represented
than by the original repetitive BLIP2 profile. Remaining errors include monitor
application icons described as physical buttons, uncertain distant text and
details, and unsupported setting/purpose judgements. The numeral3 close-up does
not establish a public/commercial setting. The candidate is qualified **with
limitations** as a caption ablation, not as factual perfection or an identified
author BLIP2 checkpoint. No caption is manually repaired.

On the exact Figure4 image, the native single-image output has69 words against
the paper example's72. It identifies the Cannery Row directory, map, rightward
red arrow, tree, green roof and trash can without the prior unsupported repeated
San Francisco string. Its raw generated-token audit passes. The original nine
development images have mean38.11 versus102.11 words and mean repeated-trigram
fraction0.32366 versus0.02023 (old/new). These are selection-set descriptive
statistics, not a population factuality or recognition-accuracy score.

Preserved failures include the15GiB offload load-time OOM when unrelated jobs
occupied the GPU, batch3/7GiB offload vision-attention OOM, and batch3
single-device vision-attention OOM on an otherwise free RTX3090. Batch2 is
separately qualified without lowering the frozen image-resolution budget.
Unrelated processes were never stopped. The earlier partial CAL1/9 report is
retained as a historical interim report, superseded for caption qualification
by the complete nine-image receipt.

The matched recognition experiment uses
`configs/approximate/detailed_caption_router_comparison_v13.json` and
`scripts/run_detailed_caption_comparison.py`. It retains every frozen native
PARSeq T1 payload and all U-Net/YOLO T3/DeepSolo expert outputs and common29/10
trained-router decisions, recording the exact source/cache hashes. Only the
caption model/profile changes. Paper scoring remains MPNet/cosine plus raw
fuzzy matching with alpha0.6/beta0.4/tau0.8; literal threshold, same-budget and
CAL-only transferred cuts remain separate. Counts are logical cached routing,
not online DeepSolo calls or end-to-end latency. The runner seals source and
runtime metadata, preserves batch membership during resume, propagates OOMs,
and has a separate native-token/RGB/reused-T1 auditor in addition to the existing
independent routing/count/semantic auditor. A completed TEST result is reported
only after every planned sample and both independent audits finish.

## Runtime qualification boundary

All five Qwen weight shards have passed their published SHA256 checks. Acquisition
resumes incomplete public model bytes with explicit HTTP range contracts and
verifies each complete shard before any inference. The initial disk acquisition
and a slower native transfer attempt remain recorded.

During development, an existing ICDAR2013 training process occupied about6.2GB
of the GPU and was left running. Qwen used explicitly recorded CPU weight
offload: the first14GiB GPU-budget reference is preserved, and the completed
15GiB GPU/32GiB CPU development run records its full device map and memory peaks.
The four overlapping reference examples produce identical complete generated
token sequences; this finite observation is not proof of global offload or
single-device equivalence. Shared-GPU generation timings are diagnostic and
cannot be used as isolated latency benchmarks.

Historical development profile: `configs/approximate/detailed_scene_captions_qwen_offload15_v5.json`.
Qualified single-image profile: `configs/approximate/detailed_scene_captions_qwen_selected_v10.json`.
Qualified batch2 profile: `configs/approximate/detailed_scene_captions_qwen_batch2_v12.json`.
The production single-image entry point is `scripts/caption_scene.py`; it requires
a separately frozen `selected_profile` and verifies checkpoint hashes before
inference. Token/provenance verification is in `scripts/audit_detailed_captions.py`;
`scripts/render_detailed_captions.py` creates the unchanged old/new image review.

## Frozen development selection

`selection_frozen.json` selects `qwen_scene_detail` before any Qwen CAL or
Figure 4 output is inspected. It identifies the main objects and useful visible
relations across all nine development images. The alternate four-to-six-sentence
profile has some worse spatial relations, while the additional literal-only
prompt misidentifies a vehicle window/doorway as a rearview mirror. All candidate
outputs are retained, including their factual errors. No profile is selected
using word-recognition F1.

The selected profile still has development limitations: the office-plaque upper
strip colors are wrong, GO GREEN lettering is described as white rather than
green, the Skytrax checker pattern is called red/white rather than black/white,
and occasional generic function/design language remains. These are model
errors, not corrected caption text. Subsequent CAL qualification accepts this
ablation with limitations; it does not establish equality to every paper example
or the unknown author checkpoint. The goal is improved concrete detail and much less invented scene
content than the previous native candidates.


The original36-caption smoke completes all54 routing policies and both audits:
72 neural cache wrappers,48 independent word-count records,72 score decisions,
36 native caption/RGB/T1-reuse checks, and full MPNet reencoding with maximum
absolute semantic error1.65e-7.81 focused unit tests pass. The production scalar
CLI also reproduces the prior69-word Figure4 caption with identical generated
token IDs in this finite placement check; allocated/reserved memory is
16,950,951,936/17,467,179,008 bytes. This does not prove global placement or
batch equivalence. Original-resolution reinspection confirms a visible wrinkled
black plastic bin liner below the dark domed lid. An initial review inferred
that the bag wording was an error from terse scene notes; that error label is
withdrawn. The unchanged96-word actual full-run caption also describes the
gray corrugated wall appearance. It describes the principal scene objects
coherently, while the paper paragraph is more precise about the rightward arrow,
curved upper banner and barrel shape. Initial and corrected review receipts
are preserved; no generated caption or selected profile changes.

The first native smoke auditor incorrectly compared the preprocessor-expanded
batch chat with an unexpanded template. Inspection of the installed pinned
Qwen processor confirms in-place image-marker expansion. The corrected audit
reconstructs each marker count from the validated visual grid; the unchanged
native outputs pass. Failed auditor source/log and correction receipt are
preserved alongside all genuine OOM evidence. No inference text was repaired.
The final report separately measures MPNet overlap against the paper paragraph;
this is semantic overlap, not a factuality or author-equivalence score. The
paper's dataset-specific fallback counts are unavailable; only this experiment's
actual cached routing decisions are reported.

The first full generation process exited with native SIGSEGV/139 after266
complete captions, without a Python traceback. No partial batch existed, and
source/configuration hashes passed. Its failure receipt and original log are
retained. The identical sealed run resumes from the next fixed batch, with
`PYTHONFAULTHANDLER=1` for diagnostics; both pending images succeed after the
restart. The native failure cause remains unresolved. No input is excluded,
no alternate caption is fabricated, and no checkpoint, precision, resolution,
prompt, seed or batch membership changes. Restart timings are not combined
into an online latency claim. Completion still requires every planned output
and both independent audits.

## Completed three-dataset recognition results

All1,304 planned captions and all1,033 official TEST images complete. Both independent audits pass:2,608 neural output wrappers,4,132 word-count records,54 policies,2,608 score decisions and1,304 native-token/RGB/reused-T1 checks. Full MPNet reencoding maximum error is7.24e-7 against the declared2e-6 tolerance.81 focused unit tests pass.

| Dataset/expert | Original BLIP2 T1/T3 F1 % | New Qwen T1/T3 F1 % | New Qwen T3-only F1 % | Common29/10 trained F1 % | DeepSolo images/total (%), each gate |
| --- | ---: | ---: | ---: | ---: | --- |
| icdar13__unet | 78.8637 | 79.6685 | 79.6748 | 84.4294 | 61/233 (26.18%) |
| icdar13__yolo | 85.7143 | 85.1125 | 86.4256 | 86.6067 | 61/233 (26.18%) |
| icdar2015__unet | 70.6177 | 75.7335 | 76.6885 | 79.0943 | 228/500 (45.60%) |
| icdar2015__yolo | 73.4558 | 76.7722 | 77.7722 | 80.8403 | 228/500 (45.60%) |
| totaltext__unet | 75.8519 | 74.1832 | 74.2141 | 79.7119 | 150/300 (50.00%) |
| totaltext__yolo | 78.9242 | 78.3071 | 78.2045 | 80.9388 | 150/300 (50.00%) |

These are punctuation-free, casefold/alphanumeric exact per-image word-multiset micro F1 results without spatial IoU, using frozen care/ignore assignments and every TEST image. Qwen improves the T1/T3 point F1 on ICDAR2015 and ICDAR2013/U-Net, while Total-Text and ICDAR2013/YOLO decrease relative to the original BLIP2 profile. The common trained router has higher observed matched-budget F1 in all six cases, including the T3-only ablation. This conditional result on historically explored benchmarks is not a population-superiority claim.

At literal tau0.8, every image requests cached DeepSolo: Total-Text82.6540%,300/300(100.00%); ICDAR2013 90.1707%,233/233(100.00%); ICDAR2015 82.7905%,500/500(100.00%), identical for U-Net and YOLO. It does not establish useful inference savings or the unknown author operating point. Actual new DeepSolo neural inferences are zero. CAL-only transferred cuts with their actual fallback counts remain separately reported in router_full/results.json.

All1,033 Qwen TEST captions have repeated-trigram fraction at most0.20, versus666/1,033 above0.20 for the original BLIP2. Dataset means are95.08/102.36/100.36 words for ICDAR2013/ICDAR2015/Total-Text. These describe length/repetition, not factual correctness. Direct review of the actual fixed19 full-run examples still finds small-color/text errors and unsupported setting/function claims; full_native_caption_visual_review.json preserves them.

The actual full-run Figure4 caption is96 words, compared with72 in the paper. MPNet overlap with the paper paragraph improves from0.607295 to0.819963. The coherent directory/map/bin/building/tree/sky description is much closer to the illustrated style, but semantic overlap is not a factuality or author-model equivalence certificate. Native batch2 produces a different paragraph from the separately verified69-word scalar output; both remain unchanged.

Actual full-run RTX3090 BF16/SDPA batch2 peaks are21,331,018,752/24,796,725,248 allocated/reserved bytes. The native SIGSEGV after266 captions was resumed from the next unchanged batch; no sample was omitted. The failure cause remains unresolved and its evidence is retained. No package was installed or upgraded, no new DeepSolo call occurred, and no commit was created.

The background finalization waiter used an outdated process-name condition after the native restart and exited before the audits existed. Once both completed audits were verified, the same reporting command was run directly. This orchestration failure changed no inference or metric output; its failed log is retained.

Final artifacts under the external experiment root: final_comparison/summary.md, summary.json, caption_review.html; router_full/results.json, per_image_scores.json, independent_audit.json and native_audit.json. Portable receipt: provenance/DETAILED_CAPTION_ROUTER_COMPLETION_V1.json. The earlier available_results_v1 report is historical and is superseded for new-caption F1 by this completed comparison.
