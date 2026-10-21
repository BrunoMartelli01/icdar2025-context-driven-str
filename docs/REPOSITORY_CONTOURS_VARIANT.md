# Opt-in repository contour localizer

`localize/repository_contours.py` reproduces the frozen author repository's
`findContours` / `boundingRect` algorithm. This is **approximate reproduction**,
not a replacement for `localize/mask_to_blocks.py` or a resolution of L-002.
See the source identity and checks in
`artifacts/diagnostics/repository-localizer-20260916/verification.json`.

| Operation | Existing component-area path | Repository variant |
|---|---|---|
| Extraction | connected components, explicit connectivity | external contours |
| Minimum area | foreground pixels | unpadded rectangle area |
| Ranking | foreground pixels | clipped padded rectangle area |
| Equal-area ties | spatial order | OpenCV contour-return order |
| Group disconnected fragments | no | no |

The upstream mask is already at **original image resolution**. This variant
does not resize it. Padding is measured in original pixels. Do not feed a
native 128x128 mask alongside a differently sized image: the crop API rejects
that mismatch. Contour order is retained for source parity and may depend on
OpenCV version; every audit records the version.

## Python use

Load parameters from the explicit approximate configuration; the existing
pipeline does not switch localizers automatically.

```python
import json
from pathlib import Path
from localize.repository_contours import RepositoryContourParams, crop_repository_blocks

config = json.loads(Path("configs/approximate/localization_repository_contours_v1.json").read_text())
params = RepositoryContourParams(**config["parameters"])
audit, crops = crop_repository_blocks(mask_original, image_rgb, params)
# audit['retained'] and crops have the same rank ordering.
```

## CPU saved-mask comparison

This compares the two methods on the **same** input mask, does not load models,
and writes a report without overwriting an existing file. Supply real paths
for `MASK`, `RGB`, and `OUT`; no dataset/checkpoint path is hardcoded.

```bash
PY=.venvs/context-str-main-py311/bin/python
CFG=configs/approximate/localization_repository_contours_v1.json
"$PY" -m diagnostics.compare_repository_localizer \
  --mask "$MASK" --image "$RGB" --config "$CFG" --output "$OUT"
```

The report contains retained/rejected boxes, area definitions, crop shapes,
configuration, input hashes, and whether the ordered boxes match. It does not
claim improved recognition accuracy. Real TRAIN comparison needs the mask
pixels, not just their hashes in compact OCR artifacts. No inference was
performed to replace missing cached masks.

## Full TRAIN comparison on PAPER

The separate `diagnostics/repository_contours_train.py` runner compares
CURRENT_ALG1, REPOSITORY_CONTOURS and PRACTICAL_REMAP_V1 on **one shared native
mask per image**, backprojected by the frozen adapter. It does not change any
pipeline default or tune parameters. The source-repository variant is not
called a paper-exact implementation.

```bash
bash diagnostics/run_repository_contours_train_host.sh /extra1/bmartelli
```

Add `--check` for a CPU-only preflight without inference or output writes.
The full run is foreground with progress. On PAPER, missing native masks are
generated once with the frozen checkpoint, then the segmenter is released
before loading PARSeq. Existing practical-v1 OCR is reused only after manifest,
shard, crop-file and runtime checks, and matching bbox plus exact RGB pixels.
Identical crops across the three methods are decoded once. No captions,
context routing, DeepSolo, training or TEST images are used.

Persistent output is `context-str-runs/repository-contours-train-v1/` under the
given storage root. Native binary masks are bit-packed into small JSON shards;
new crops are implicit bbox ROIs into the verified original images rather than
additional PNG copies. Masks and per-image OCR checkpoints use atomic writes
and contract/payload checksums, so rerunning the command resumes completed
images. A partially completed OCR image may be recomputed after interruption.
Changing code/config/inputs requires a new output directory, set explicitly
using `REPOSITORY_CONTOURS_OUTPUT`.

The report includes geometry (IoU .25/.5/.75, one-to-one proposal P/R/F1),
casefold exact/fuzzy OCR out of all 849 TRAIN words, wordness, per-crop/per-word
records and the number of masks matching the historical v1 cache. All three
reported methods always use identical current masks, irrespective of historical
mask equality. OCR-to-GT association is post-hoc maximum IoU with positive
intersection required; it is a diagnostic, not end-to-end Table 5 accuracy.

Scientific choices: frozen repository parameters are REPOSITORY_SPECIFIED;
current connectivity/spatial ties and post-hoc matching are existing ASSUMPTIONs.
The official manifest pins RGB/BMP but omits some TXT checksums: those are
recorded from disk in the first run's protocol and checked on every resume,
not claimed to have independently verified historical hashes. Geometry and
source preprocessing are unchanged. Float32, batch-one sequential model use
and cumulative GPU memory peaks are recorded; this is approximate reproduction.

## User-requested native128 area and padding variant

`localize/native_repository_contours.py` runs the same external-contour code on
the verified native128 binary mask without resizing. It applies **bbox area
150 native pixels squared**, **padding 6 native pixels**, sorting and top10
before mapping the padded boxes to original RGB coordinates (floor lower,
ceil upper edges). This coordinate frame is an explicit ASSUMPTION requested
by the user, not an established paper requirement. The older variants remain
untouched. Native128 foreground-pixel area and bbox area must not be confused.

CPU geometry is evaluated by `diagnostics/repository_contours_native_train.py`
using the previous run's hash-verified native mask shards. Persistent output:
`artifacts/diagnostics/repository-contours-native128-train-v1/geometry.json`.
No model inference, CUDA, mask resize or OCR is used for geometry.

Optional new-crop PARSeq can be run manually, foreground, on PAPER:

```bash
bash diagnostics/run_repository_contours_native_ocr_host.sh /extra1/bmartelli
```

The wrapper reuses geometry if available and never runs segmentation. It writes
separate resumable OCR shards and `ocr-results.json` in the same directory,
without replacing geometry or previous comparison reports. Reuse requires the
same qualified PARSeq artifacts; candidate selection remains GT-independent.
