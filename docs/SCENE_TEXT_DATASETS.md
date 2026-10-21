# Additional scene-text datasets

This repository supports checksum-frozen ingestion of ICDAR2015, Total-Text,
and CTW1500 through `scripts/prepare_scene_text_dataset.py`. This is an
annotation/identity milestone, not a Table 5 result.

## Scientific boundary

| Dataset | Paper relationship | Annotation granularity | Classification |
|---|---|---|---|
| ICDAR2015 | Table 5 recognition benchmark | word quadrilateral | `PAPER_SPECIFIED` |
| Total-Text | segmentation and Table 5 benchmark | word polygon | `PAPER_SPECIFIED` |
| CTW1500 | absent from the paper | curved text line | `ASSUMPTION` / practical extension |

Recognition matching and accuracy remain `UNRESOLVED` under E-001. The current
downloads were physically checked as 1,000/500 ICDAR2015 images, 1,000/500
CTW1500 images, and 1,255/300 Total-Text images. The training path below is a
`PRACTICAL_APPROXIMATION`, not a strict reproduction of Table 3 or Table 5.

CTW1500 requires two distinct artifacts.  The frozen MMOCR mirror remains the
source of images and detection polygons used for segmentation.  Text-line
transcriptions come from the upstream `ctw1500_e2e_annos.zip`; they must never
silently replace the segmentation source.  Download and qualify them with:

```bash
bash scripts/prepare_ctw1500_recognition_host.sh /extra1/bmartelli
```

The qualifier requires 1,000 TRAIN and 500 TEST annotation files, rejects
detection-only archives, verifies every annotation checksum, and matches every
physical recognition instance to the frozen detection geometry at bbox IoU >=
0.50 while reporting the 0.75/0.90/0.95 counts. The frozen upstream archive has
eight TRAIN header/count mismatches, eight detection regions absent from its
recognition files, and six regions without a transcription. These are recorded;
no text or geometry is invented, and missing transcriptions are excluded from
recognition scoring. The recognition labels also restore 403 TRAIN `###` flags
that are absent from the MMOCR mirror metadata, so the already-trained CTW
segmentation model retains a documented target-policy limitation.
It writes `ctw1500_recognition_{train,test}.json` under the prepared manifest
directory.  CTW1500 remains text-line-level and out of the paper; its exact
score is therefore not a word-accuracy result.  Existing V1/PARSeq outputs can
be rescored without GPU inference using
`scripts/rescore_ctw1500_v1_parseq_host.sh`.

## Persistent on-disk layout

Keep archives and extracted data outside Git, under:

```text
/extra1/bmartelli/datasets/context-str/
  icdar2015/raw/{train_images,train_gt,test_images,test_gt}/
  total_text/raw/{train_images,train_gt,test_images,test_gt}/
  ctw1500/raw/{train_images,train_gt,test_images,test_gt}/
```

The qualifier accepts explicit directories rather than guessing a layout. For
example:

```bash
python -m scripts.prepare_scene_text_dataset build \
  --dataset icdar2015 --split test \
  --data-root /extra1/bmartelli/datasets/context-str/icdar2015 \
  --images /extra1/bmartelli/datasets/context-str/icdar2015/raw/test_images \
  --annotations /extra1/bmartelli/datasets/context-str/icdar2015/raw/test_gt \
  --output manifests/icdar2015_recognition_test.json
```

Omit `--allow-partial` for official qualification. The generated manifest
records every image/annotation SHA256, dimensions, instance counts, care counts,
parser revision, granularity, and scientific classification. `verify` reparses
all annotations and fails on any byte drift.

Acquisition sources and expected counts are frozen in
`configs/approximate/datasets/scene_text_sources_v1.yaml`. Dataset archives and
extracted bytes must not be committed.

## Segmentation preparation and four training runs

`scripts/prepare_scene_text_training_host.sh` creates one atomic prepared tree:

```text
/extra1/bmartelli/datasets/context-str/scene-segmentation-v1/
  datasets/{ctw1500,icdar2015,total_text}/{train,test}/
  manifests/
  qualification.json
```

The preparer keeps downloaded archives unchanged. It extracts CTW1500 and
ICDAR2015 images once, hard-links Total-Text images where the filesystem allows
it, writes lossless binary target/validity PNGs, and records every SHA256.
ICDAR2015 word quadrilaterals and CTW1500 line polygons are filled to construct
segmentation targets (`ASSUMPTION`). Total-Text uses its distributed pixel masks,
thresholded at 128 to remove JPEG encoding noise. Ignore polygons produce zero
BCE validity. Border points outside the pixel extent are clipped and counted in
each manifest. Six EXIF-rotated CTW1500 images are normalized losslessly to PNG
so masks remain aligned.

Prepare once on PAPER:

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str && bash scripts/prepare_scene_text_training_host.sh
```

The four configs preserve the author-clarified AG-U-Net, frozen ResNet-50,
256x256 image / 128x128 target, Adam 1e-4, weight decay 0.01, batch 8, masked
BCE, flips/rotations, 100 epochs, and training-fgIoU checkpoint selection:

| Variant | Unique train images | Sampling |
|---|---:|---|
| `ctw1500` | 1000 | deterministic epoch shuffle |
| `icdar2015` | 1000 | deterministic epoch shuffle |
| `total_text` | 1255 | deterministic epoch shuffle |
| `combined` | 3255 | deterministic domain-balanced oversampling, 1255/domain |

Run one foreground training at a time:

```bash
bash scripts/train_scene_text_segmentation_host.sh ctw1500
bash scripts/train_scene_text_segmentation_host.sh icdar2015
bash scripts/train_scene_text_segmentation_host.sh total_text
bash scripts/train_scene_text_segmentation_host.sh combined
```

Before the full runs, qualify each physical GPU path using the same command with
`--qualify`; qualification outputs use a separate root and cannot overwrite a
full run:

```bash
bash scripts/train_scene_text_segmentation_host.sh ctw1500 --qualify
bash scripts/train_scene_text_segmentation_host.sh icdar2015 --qualify
bash scripts/train_scene_text_segmentation_host.sh total_text --qualify
bash scripts/train_scene_text_segmentation_host.sh combined --qualify
```

An interrupted run is resumed by passing its checkpoint as the second argument.
No configuration uses test images for training or checkpoint selection.
