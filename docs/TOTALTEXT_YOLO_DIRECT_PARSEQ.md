# Direct YOLO boxes → PARSeq on Total-Text

This **approximate reproduction** diagnostic reads the already frozen YOLOv8s
word-object detections, crops each stored integer xyxy rectangle from the
original PIL RGB raster, and runs the qualified genuine PARSeq adapter. No
padding is added. Each box produces one reading, including empty readings.
The three previously declared YOLO thresholds (0.05, 0.25, 0.40) are evaluated
using the same set of new OCR readings. Masks and grouping do not select the
direct crops. GT is read only after all OCR shards are frozen.

The experiment reuses the qualified `parseq-bb5792a6.pt` checkpoint, pinned
runtime archive, official RGB 32×128 bicubic preprocessing, autoregressive
decoding and one refinement iteration. That checkpoint is not established as
the paper's PARSeq-S; this is an external detector comparison, not STRICT mode.
Paper sections 3.3/3.5 and the existing PARSeq adapter were inspected before
implementation. See `TT-YOLO-DIRECT-PARSEQ-V1` in `PAPER_AMBIGUITIES.md`.

The primary result is micro precision, recall and Hmean over all 300 TEST
images and all 2,204 frozen care GT words. Matching requires equal text after
Unicode casefold and P/S removal, with original-image bbox IoU≥0.5 and a
maximum one-to-one matching. Whitespace is preserved. Duplicates and empty or
symbol-only readings remain in the prediction denominator. Ignored GT words
were already excluded by the frozen source protocol. This is the same local
bbox proxy used by previous tests, not the official polygon spotting Hmean.
Detector-only geometry matches and empty readings are also reported.

## Measured full TEST result

The genuine CPU FP32 run completed all 300 images and 8,318 direct crops.
All 2,204 care GT normalize to nonempty words. No sample failed or was omitted.

| YOLO threshold | Outputs | Correct | Precision | Recall | Local Hmean | Normalized empty outputs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 8,318 | 1,833 | 22.04% | 83.17% | 34.84% | 430 |
| 0.25 | 3,224 | 1,688 | 52.36% | 76.59% | 62.20% | 77 |
| 0.40 | 2,367 | 1,574 | 66.50% | 71.42% | 68.87% | 36 |

At threshold0.40, direct crops produce125 more correct words than the previous
YOLO slot assignment (1,449/2,042 outputs, Hmean68.25%). The increase in correct
words accompanies325 more outputs, reducing precision but slightly increasing
Hmean. The compact DeepSolo-slot reference remains at1,499/1,886 outputs and
Hmean73.30%. At threshold0.05 the direct boxes exceed the original frozen crop
selection by88 correct words (1,833 versus1,745), with fewer output crops.

The recorded per-image OCR processing sums to171.39seconds, including the first
PARSeq load; this is a CPU measurement, not GPU timing. It excludes the earlier
YOLO detection run and the subsequent GT evaluation. FP32 batch32, maximum
input `[32,3,32,128]`, four CPU threads and seed0 are recorded in the plan.
CUDA peak fields are null because new recognition was executed on CPU.

Eight focused tests pass. The genuine smoke and a cached smoke replay completed.
An independent SciPy sparse maximum bipartite matching audit reproduced all
recognition and geometry counts and checked every original crop's pixel hash,
box coordinates, detector score and source image identity across all300 images.
The run preserves a source snapshot and per-image OCR/evaluation shards.

Full report: `/tmp/yolo-direct-parseq/full-cpu-v1/results.json`,
SHA256 `cea2c557adcede16d0141d99dc5ced381ebd2f8215631b3c68cf7becb968d3b5`.
Independent audit: `/tmp/yolo-direct-parseq/full-cpu-v1/audit.json`.

All detector shards, manifest, images, PARSeq weights and source are identified
by hashes; each new OCR shard retains box, detector score, crop pixel hash,
raw text and token probabilities. A cached run refuses changed contracts.
The device is explicit. CPU FP32 is a `HARDWARE_DEVIATION`, and does not
provide a GPU speed or VRAM qualification. GPU errors/OOM are recorded and
propagated. Batch size and CPU thread count are in the approximate config.

For the existing detector run, launch on a GPU host with:

```bash
bash scripts/test_totaltext_yolo_direct_parseq.sh --detector-run /tmp/yolo-word-slots/evaluation/full --output /tmp/yolo-direct-parseq/full-gpu
```

For an explicit CPU run, add `--device cpu` and use a different output folder.
For one-image genuine smoke, also add `--limit 1`.
