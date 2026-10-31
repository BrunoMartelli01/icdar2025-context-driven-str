# Matched token-router training on qualified U-Net and YOLO

**Approximate reproduction / user-authorized research extension.** The user
requested training the existing token router on the best available U-Net and
reporting threshold-dependent word F1, image routing counts and a YOLO comparison.
Paper Sections3.2–3.3/Algorithm1 and3.5/Algorithm2, existing U-Net/PARSeq,
token-router and qualified DeepSolo implementations were inspected first.
This is a trained router extension, not the paper's training-free MPNet/fuzzy rule.

## Backbone selection and retained pipeline

The completed symbol-free word-slot study selects `pixel_pil` as its strongest
standalone calibration control and it also leads the reused TRAIN125 evaluation.
Its underlying model is the qualified **Text+Kernel ResNet50 Attention U-Net**:
`adaspot-gap-v1-20261015T125256Z/control/best.pt`, selected at continuation epoch2,
SHA256 `36cd330376b53b5df6a717025bc59be2320b4879250b46b275ec526822e030ca`.
It continues the original epoch61 Text+Kernel checkpoint; it is not the unchanged
epoch61 state. The complete checkpoint is loaded strictly without missing keys.

Selection evidence is `word-slot-alphanumeric-20261025-v1/full_v2/`:
standalone calibration100 F1=69.2362%, and reused TRAIN125 F1=60.3277%, each with
0/100 or0/125 images sent to DeepSolo (0%), under that source's localized metric.
These source scores are not the new harmonized lexical metric. "Best" means
the strongest qualified control in this completed validation comparison, not a
proven global optimum or a new TEST-based choice.

Keep the source's pixel-PIL pipeline: centered letterbox RGB512/ImageNet,
native256 Text>=0.65/Kernel>=0.5, kernel BFS instance expansion, area ordering,
at most20 OCR crops, original RGB PIL crops and qualified PARSeq AR/refine1.
The existing TRAIN-selected full sequence probability threshold is0.7, including
EOS. Character-only products would be a different filter. Mean text probability
over each component supplies the router's detector-confidence feature. No
segmentation or recognition weights are updated.

## Matched router experiment

- Complete usable TRAIN1253, after the same two exact TRAIN/TEST RGB exclusions;
  complete TEST300 remains separate. Group remaining TRAIN RGB duplicates.
- U-Net and YOLO receive identical token architecture, seed20280970, five folds,
  optimizer and5/10/20 checkpoint grid. Select epoch using TRAIN OOF combined
  micro F1 at50% bypass; refit each selected head on all1253 TRAIN images.
- Use Unicode casefold then `isalnum` for both expert/GT matching and weak
  spatial word labels. Primary matching is per-image lexical multiset, no IoU;
  duplicates and every care denominator remain. Empty normalized strings cannot
  match. Native256 evaluation-only validity follows the earlier YOLO protocol.
- Use the same improved native ViTAEv2-S/TextOCR DeepSolo profile, detection
  threshold0.35 and recognition confidence0.925. Its TEST cache is qualified;
  export missing TRAIN1128 outputs with the same native resize1024/1892 and
  inference floor0.05 in the existing isolated legacy runtime. No TEST reselection.
- Reuse qualified CLIP image patches and BLIP2 image-only query arrays by exact
  image/RGB/NPZ/schema identity. The BLIP query method depends only on RGB.
  Regenerate U-Net word/crop CLIP arrays and statistics. Requalify unchanged
  global CLIP tokens on a real sample; do not rerun BLIP language likelihood for
  this token-only experiment. Foundation models stay frozen.
- Fit scalar thresholds on selected TRAIN OOF score quantiles and transfer the
  actual cuts to TEST; retain ties and report actual rates. Also evaluate a fixed
  utility-cut grid. Keep these separate from descriptive equal-quota batch
  ranking and the privileged same-budget GT oracle.

Every reported policy includes DeepSolo images/total and percentage, plus base
and all-DeepSolo endpoints. These are logical routing decisions on cached outputs,
not measured online latency. Actual new TRAIN fallback inference is logged
separately. The source's historical TEST exploration and inherited backbone
training exposure remain explicit.

## Runtime, qualification and status

Only existing environments are used. Actual PAPER GPU is RTX3090/24GB. FP32
stages run sequentially, PARSeq runs on CPU with batch32, token attention uses
deterministic math attention and TF32off. Native DeepSolo retains its separately
qualified cuDNN settings. Record allocated/reserved GPU peaks, shapes, model
state hashes, source snapshots and explicit OOM/failure artifacts. These choices
are HARDWARE_DEVIATION; no8GB qualification or speed claim is implied.

Configuration: `configs/approximate/unet_token_router_v1.json`; explicit host
paths: `configs/approximate/unet_token_router_local_paths_v1.json`.
Runner: `scripts/run_unet_token_router.py` and
`scripts/unet_token_router_stages.py`; crop/cut boundaries:
`recognition/unet_token_routing.py`.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 .venvs/context-str-main-py311/bin/python -u -m scripts.run_unet_token_router \
 --mode approximate --config configs/approximate/unet_token_router_v1.json \
 --paths configs/approximate/unet_token_router_local_paths_v1.json --smoke
```

Remove `--smoke` for full execution. `--preflight` validates without inference;
to resume, invoke `--run-root /absolute/sealed/root` from that run's
`code_snapshot` directory with the same directory on `PYTHONPATH` and the existing
main Python interpreter. This explicitly selects the executed source snapshot.
Predeclaration claimed no performance result; completion and audited results
are recorded below.

An initial helper test fixture omitted the qualified character log-probability
field and failed explicitly; the fixture was corrected to the real schema.
Inspection also established the required EOS-inclusive sequence filter before
any GPU experiment. Preserve failed checks as qualification history; no source
model output was changed to make a check pass.

Scientific choices: `UNET-TOKEN-ROUTER-001` in
[PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md).

## Qualification and completed full run

The first GPU smoke (`78de5a562116`) completed frozen expert/context stages and
then failed explicitly at router training because the qualified feature loader
returns `(arrays, metadata)`, not just an array dictionary. The interface was
corrected, with a regression check for independent U-Net/YOLO words and shared
image-only tokens. No model outputs were substituted.

The corrected smoke (`32e9a71dde88`) passed both matched fits, final evaluation
and188 independent checks. The focused suite passed47 tests. Maximum allocated/
reserved memory in smoke: U-Net/PARSeq GPU stage564,542,976/631,242,752 bytes
(PARSeq itself on CPU); frozen CLIP752,758,272/845,152,256 bytes. These are smoke
measurements, not full-population peaks or latency.

The full1253/300-image experiment completed at external artifact root
`unet-token-router-v1/full/0cdd1aa41bdf`. The immutable code snapshot protects it
from concurrent repository changes. Actual native DeepSolo inference ran
for1128 missing TRAIN samples; all other fallback outputs came from qualified
caches. Every final head was refitted on all1253 usable TRAIN images.

Derived report exporter: `scripts/report_unet_token_router.py`. Its first report
attempt explicitly failed because the main environment has no Matplotlib; the
existing YOLO environment provides it, without installation or dependency changes.
The smoke report was successfully regenerated separately in `report_v2`, preserving
the failed partial output. Rendering does not refit networks or change inference.
Subsequent smoke report qualification also verified original inputs, generated
stage receipts, executed source and learned checkpoint hashes.

## Audited results

Completion: **SUCCESS**; independent word/count/route/model-scope checks: **6366 PASS**. The derived report rechecked17843 original, generated and snapshot hashes. Both final heads contain220035 parameters and fit all1253 usable TRAIN images. TRAIN OOF selects5 epochs for U-Net and10 for YOLO; TEST does not select epochs or cuts.

The frozen U-Net source replays225 prior TRAIN calibration/evaluation images with identical original crop bounds, raw transcripts and selected words. Maximum sequence-probability difference is1.7524e-5 for CPU versus historical qualified PARSeq inference; every selected row remains identical. The invoked scientific code matches the qualified smoke source.

Portable numerical summary and identities: [completion manifest](../provenance/unet_token_router_20261027_completion.json). Complete external report: [all cuts and quotas](/extra1/bmartelli/context-str-artifacts/unet-token-router-v1/full/0cdd1aa41bdf/report/report.md); [132-row CSV](/extra1/bmartelli/context-str-artifacts/unet-token-router-v1/full/0cdd1aa41bdf/report/threshold_results.csv); [curves](/extra1/bmartelli/context-str-artifacts/unet-token-router-v1/full/0cdd1aa41bdf/report/routing_curves.png).

### Standalone endpoints

| Split | Expert | F1 (%) | DeepSolo images/total (%) |
| --- | --- | ---: | ---: |
| TEST | UNET + PARSeq | 63.0532 | 0/300 (0.00%) |
| TEST | YOLO + PARSeq | 74.2581 | 0/300 (0.00%) |
| TEST | Improved native DeepSolo | 82.6540 | 300/300 (100.00%) |
| TRAIN | UNET + PARSeq | 74.0913 | 0/1253 (0.00%) |
| TRAIN | YOLO + PARSeq | 71.5368 | 0/1253 (0.00%) |
| TRAIN | Improved native DeepSolo | 82.9506 | 1253/1253 (100.00%) |

### Equal acceptance quotas

These descriptive batch ranks use no GT. The DeepSolo column applies to both learned methods and their separate GT oracles at the same quota. TRAIN is OOF; TEST uses final full-TRAIN heads. These are lexical micro F1 values, without localization IoU, under the declared symbol/ignore/empty rules.

| Split | Accepted light images | DeepSolo images/total (%) | U-Net F1 (%) | YOLO F1 (%) | U-Net oracle F1 (%) | YOLO oracle F1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TEST | 30/300 | 270/300 (90.00%) | 82.0171 | 82.7570 | 83.7819 | 84.3735 |
| TEST | 90/300 | 210/300 (70.00%) | 80.6807 | 81.7966 | 83.8050 | 84.8414 |
| TEST | 150/300 | 150/300 (50.00%) | 78.1122 | 80.9558 | 82.5465 | 84.2983 |
| TEST | 210/300 | 90/300 (30.00%) | 73.6970 | 78.5410 | 79.0323 | 82.2439 |
| TRAIN | 125/1253 | 1128/1253 (90.02%) | 82.6329 | 82.4221 | 84.9610 | 84.4087 |
| TRAIN | 376/1253 | 877/1253 (69.99%) | 82.1006 | 81.0716 | 85.6850 | 84.5154 |
| TRAIN | 626/1253 | 627/1253 (50.04%) | 81.1063 | 79.2572 | 85.6850 | 83.5215 |
| TRAIN | 877/1253 | 376/1253 (30.01%) | 79.6426 | 76.7703 | 84.4320 | 80.9667 |

### Scalar cuts transferred from TRAIN OOF

Higher raw predicted utility favors the light expert; it is not a probability. Every cut below was fixed using TRAIN OOF. The table reports actual TEST acceptance, preserving ties. The full CSV also includes the predeclared fixed utility grid and all TRAIN quantiles.

| Expert | TRAIN target | Raw cut | Accepted light TEST | DeepSolo images/total (%) | TEST F1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| UNET | 10% | 0.760365 | 72/300 | 228/300 (76.00%) | 81.0345 |
| UNET | 20% | 0.284693 | 159/300 | 141/300 (47.00%) | 77.5609 |
| UNET | 50% | -0.875037 | 257/300 | 43/300 (14.33%) | 69.9321 |
| YOLO | 10% | 0.783076 | 6/300 | 294/300 (98.00%) | 82.6818 |
| YOLO | 20% | -0.087638 | 20/300 | 280/300 (93.33%) | 82.6768 |
| YOLO | 50% | -2.185116 | 134/300 | 166/300 (55.33%) | 81.1361 |

### Interpretation and score calibration diagnostic

At150 accepted TEST images, the learned U-Net gate yields78.1122% F1 versus YOLO80.9558%; both send150/300 images to DeepSolo (50.00%). The paired U-Net-minus-YOLO difference is-2.8436pp with conditional RGB-cluster95% interval[-4.8148,-0.9551]pp. This conditions on fixed routes and excludes variation from training seeds.

At that same150/300 (50.00%) DeepSolo cost, the U-Net GT oracle reaches 82.5465%, below the82.6540% all-DeepSolo endpoint with300/300 (100.00%) DeepSolo images. The corresponding YOLO oracle is84.2983% with150/300 (50.00%) DeepSolo images. Improving this U-Net gate alone cannot beat the all-DeepSolo endpoint while retaining exactly half of TEST with these experts and eligibility constraints.

The raw U-Net OOF cut for50% TRAIN acceptance is-0.875037. It accepts 626/1253 OOF images, sending627/1253 to DeepSolo (50.04%). On the final head's in-sample TRAIN predictions it accepts1114/1253, sending139/1253 (11.09%); on TEST it accepts257/300, sending43/300 (14.33%), with69.9321% F1. Thus raw-score calibration already changes across OOF and full-TRAIN refitting; the acceptance drift is not isolated to TEST. This is a posthoc diagnostic, not a new threshold-selection rule. No TEST recalibration was applied. Equal-quota ranks provide the predeclared comparison with controlled routing counts.

The U-Net standalone TRAIN/TEST gap and its optimistic final scores are observations. Their individual causal contributions require further controlled experiments. The results preserve the existing TRAIN selection and historical TEST/expert-exposure qualifications.

### Full-run memory and reproducibility

Actual hardware: RTX3090/24GB. FP32 models execute in separate stages; PARSeq uses the CPU. U-Net GPU peaks across TRAIN/TEST:564542976 allocated / 631242752 reserved bytes, batch1, RGB512 input and two native256 maps. Frozen CLIP maximum peaks:752758272 /847249408 bytes, native224 inputs with configured image batch8. Final head fitting peaks: U-Net107076608 / 121634816 bytes, YOLO140251136 /163577856 bytes, configured batch8 with980 patch tokens,32 cached BLIP queries and padded word tokens. These are final-fit measurements, with intermediate CV measurements retained in each checkpoint.

New native DeepSolo TRAIN inference ran at batch1, FP32,1024/1892 resize; measured maxima over1128 outputs are2117447680 allocated /4102029312 reserved bytes (example total_text:1378). The raw outputs retain per-image shapes, timings and memory. There are no full-run failure or OOM records. Actual new TEST DeepSolo inference is0: its300 qualified outputs are reused. All policy counts above describe logical routing on these cached experts, not measured online latency or a fresh8GB hardware qualification.

To regenerate the derived report with the existing plotting environment:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
 .venvs/context-str-yolo-py311/bin/python -m scripts.report_unet_token_router \
 --run-root /extra1/bmartelli/context-str-artifacts/unet-token-router-v1/full/0cdd1aa41bdf \
 --output /extra1/bmartelli/context-str-artifacts/unet-token-router-v1/full/0cdd1aa41bdf/report
```
