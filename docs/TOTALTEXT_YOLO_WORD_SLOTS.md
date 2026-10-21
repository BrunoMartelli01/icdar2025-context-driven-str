# Lightweight word-object slots for frozen Total-Text crops

This is an **approximate reproduction** experiment outside the paper's strict
segmentation localization and DeepSolo fallback. Each word is an object of one
class (`text`) with an axis-aligned bounding box. YOLO supplies positions only;
the existing frozen PARSeq text and GT-blind crop-to-slot assignment remain in
use.

## Candidate and limits

The candidate is [`yolov8_5k.pt`](https://github.com/Alfadhils/YOLOv8-CRNN-Scene-Text-Recognition/blob/bc195de0ec8a470eaa026c004820a7c275663cbf/checkpoints/yolov8_5k.pt),
a 22,484,953-byte YOLOv8s object-detector checkpoint in the author's repository.
The author reports training on 5,000 TextOCR images with word boxes and
67.559% mAP@0.5 on its own validation split. This is **not** a Total-Text
result. The training conversion [excludes one-character transcriptions](https://github.com/Alfadhils/YOLOv8-CRNN-Scene-Text-Recognition/blob/bc195de0ec8a470eaa026c004820a7c275663cbf/src/yolov8_datagen.py),
which may reduce recall on short Total-Text words. YOLOv8s has 11.2 million
parameters at 640 input according to [Ultralytics' model table](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md).

The checkpoint is pinned by repository commit
`bc195de0ec8a470eaa026c004820a7c275663cbf` and Git blob SHA1
`a3abecbbc2efa5c5fde0c417600816a48af1bfa8`. The evaluator checks both
size and Git blob identity, then records a local SHA256. The local file matches:
SHA256 `b9f153febb1de6a91a9b771e66a5a4b404bc78111b9c7ee3f18cbf1a9379c7af`.
Static inspection of its PyTorch archive (without loading the pickle) finds
one `text` class, detection task, one class in the head, and saved Ultralytics
version `8.1.0`. This is the runtime version pinned for the test. A generic
COCO YOLO checkpoint is not an equivalent word detector.

## Completed Total-Text TEST diagnostic

The GPU-host run completed on all 300 TEST images and 2,204 GT words with
Ultralytics 8.1.0 and PyTorch 2.4.1+cu124. Each detector, selection and
evaluation stage has 300 per-image shards, and there is no failure file. An
independent shard-sum audit exactly matches the aggregate report. The frozen
PARSeq selection contains 1,745 correct words among 16,678 output crops.

| Detector and control | Slots | Final outputs | Correct words | GT box matches | Local Hmean proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO, score 0.05, crop IoU 0.10 (most correct) | 8,318 | 5,678 | 1,690 | 2,036 | 0.429 |
| YOLO, score 0.40, crop IoU 0.50 (best Hmean) | 2,367 | 2,042 | 1,449 | 1,691 | 0.683 |
| DeepSolo, score 0.40, crop IoU 0.50 reference | 2,040 | 1,886 | 1,499 | 1,812 | 0.733 |

At the permissive YOLO control, 1,736 correct original crops overlap at least
one slot, while final one-to-one assignment retains 1,690. The detector's best
box recall is 2,036/2,204 (92.4%), compared with 2,137/2,204 (97.0%) for
DeepSolo. These are **approximate reproduction** diagnostics with a local
normalized-word, axis-aligned-box Hmean proxy, not the official Total-Text
polygon Hmean or a strict paper reproduction.

The run used an NVIDIA RTX 3090, a `HARDWARE_DEVIATION` from the target RTX
3070. Measured peak PyTorch CUDA allocation was 115,851,776 bytes and peak
reservation was 144,703,488 bytes for FP32 YOLO inference at input size 640,
batch 1. These figures do not qualify the 8 GB target GPU.

## Launch

The one-command launcher uses the existing main environment's PyTorch and
dependencies read-only, installs pinned YOLO dependencies into a separate YOLO
environment if needed, runs focused tests, then a one-image smoke and the
complete TEST diagnostic. It requires a GPU host with PyPI access for initial
installation:

```bash
bash scripts/run_totaltext_yolo_word_slots.sh /tmp/yolo-word-slots/yolov8_5k.pt
```

Local preflight verified the 300-image TEST manifest, the first source image
checksum, frozen PARSeq selection, checkpoint identity and color conversion.
The 41 focused tests pass. The restricted Codex session could not access CUDA
or resolve PyPI; the later GPU-host run supplied the reported results.

The completed results are in `/tmp/yolo-word-slots/evaluation/smoke/results.json` and
`/tmp/yolo-word-slots/evaluation/full/results.json`. To use another output
location, pass it as the second argument. The loader converts the stored PIL
raster to a BGR NumPy array, as required by [Ultralytics 8.1.0 prediction
preprocessing](https://github.com/ultralytics/ultralytics/blob/v8.1.0/ultralytics/engine/predictor.py).

The result contains detector-only GT box coverage, correct PARSeq crops still
covered, final selected correct words, output count and local Hmean proxy at
predeclared score/crop-overlap controls. All 300 images and 2,204 GT words are
kept in the complete denominator. GT is opened only after YOLO predictions
and GT-blind assignments are frozen. The metric normalizes case and strips
Unicode punctuation/symbols, with one-to-one bbox IoU ≥ 0.5. It is not the
official Total-Text polygon Hmean.

The comparison reference is
`/tmp/totaltext-independent-detector-slots-v3/full/results.json`: DeepSolo
produced 12,469 low-threshold slots with 2,137/2,204 GT box coverage, while
2,040 high-score slots covered 1,812/2,204. The compact DeepSolo arm gave
1,499 correct PARSeq words among 1,886 outputs, local Hmean proxy 0.733.
