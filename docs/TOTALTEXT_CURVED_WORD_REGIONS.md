# Total-Text curved word-region training (practical approximation)

This experiment retains the existing Attention-Gated U-Net, pretrained/frozen
ResNet-50 encoder, 256×256 RGB preprocessing, 128×128 target, Adam/BCE protocol,
and official 1255-image Total-Text TRAIN split. It changes **only the training
target**, with a second variant that also changes the pixel loss weighting.
Neither variant is a strict reproduction of the paper's pixel-level target.

The original Total-Text word polygon is filled as a curved ribbon, not replaced
by an axis-aligned rectangle. Overlapping polygon pixels are assigned to the
nearest pixel exclusive to one participating word. A black 8-neighbour
separator is removed where different assigned words touch. Ignore polygons
remain zero-weight through the validity mask. A per-word uint16 owner map is
retained for audit. The same deterministic geometric augmentation is applied
to the owner map, and a separator is enforced again after 128×128 resizing;
otherwise bilinear resizing can reconnect distinct words.

Preparation audit: 1255 TRAIN images, 9285 care words, 5195 curved polygons,
493 images with source-polygon overlaps, zero vanished source-resolution
words. The target covers 99.23% of the existing pixel-mask foreground over
valid pixels. Without the native-resolution owner separator, 679 images have
touching different-word labels after resize; with it, zero do. See
`qualification.json` and `qualification_native128.json` for machine-readable
records. The 128×128 target still loses foreground for 26 of 9285 very small
word regions; this is a known resolution limit, not hidden by the audit.
These are target properties, not trained-model results.

For manual inspection,
`artifacts/totaltext-curved-word-regions-v1/totaltext-train-worst50-gt-comparison.zip`
contains the 50 worst TRAIN cases, selected by original pixel-GT coverage
and retained polygon area. It includes exact old/new/separator PNGs, an RGB +
old + new + difference preview per image, and a per-image manifest. Preview
JPEGs are visual aids only; evaluate from the lossless PNGs.

The two frozen configurations use identical images, word targets, splits,
augmentations, initialization protocol and checkpoint selection:

| Variant | Separation-pixel BCE weight |
| --- | ---: |
| `standard` | 1.0 |
| `fusion` | 4.0 |

The fusion variant upweights valid zero-target pixels in an auxiliary
inter-word separator map. The map includes source contact pixels, short gaps
reached from two words within one native-128 pixel radius, and contacts
detected after downsampling. It does not add a new network head or use OCR/GT
text. The weight `4.0` is an untuned `ASSUMPTION`, not a paper parameter.

Prepared target root:
`artifacts/totaltext-curved-word-regions-v1/`.
Its `qualification.json` records counts, source-polygon overlap and foreground
removed by separation. The manifest pins every image, target, validity,
separator and owner-map SHA256. The full source images are hard-linked when
possible, so they do not consume a second copy of the image bytes.

On PAPER, from the repository root, run sequentially:

```bash
bash scripts/train_totaltext_curved_word_regions_host.sh standard
bash scripts/train_totaltext_curved_word_regions_host.sh fusion
```

Each run has a distinct name and writes under
`artifacts/totaltext-curved-word-regions-v1-runs/`. Training is foreground,
prints epoch progress, and records checkpoints plus peak GPU memory. These
commands are not an evaluation: after training, compare target-native 128×128
foreground IoU/F1, word connected-component topology, Algorithm 1 proposals,
and PARSeq recognition on the held-out TEST split. Do not infer improvement
from TRAIN foreground IoU alone.

## Tighter curved target: refined-v2

Adding vertices along an unchanged polygon edge does not change its mask. The
separate `refined-v2` target instead uses the official distributed pixel-level
Total-Text TRAIN mask to contract each filled v1 word polygon around its actual
text pixels. It grows the smallest connected distance envelope inside each
v1 owner (at least one native-128 pixel in source scale), retaining every
pixel-level GT seed. It never changes word ownership, overlaps adjacent words,
or uses transcription/TEST labels. If a word has no seed, its polygon is
disconnected, or the tighter raster worsens native-128 connectivity, the
corresponding v1 owner is restored. A whole-image v1 fallback handles rare
bilinear interactions among adjacent word owners. These rules were declared
without model inference or recognition-based selection.

The prepared TRAIN set has 1255 images and 9285 care words. It retains all
39,942,201 source pixel-GT seeds within v1 word owners. Refined foreground
occupies 86,617,229 pixels versus 104,843,104 for v1 (17.38% less filled
area), with 103 word-region restorations by the native connectivity guard.
Native-128 QA across every TRAIN word finds zero worsened connected-component
counts: v1 has 26 absent / 9219 single / 40 multi-component words, refined-v2
has 22 absent / 9225 single / 38 multi-component words. These counts refer to
the no-augmentation training transform, not predictions from a trained model.
These figures describe **training labels only**, not segmentation accuracy.
The two isolated configurations are `refined` (separator weight 1.0) and
`refined-fusion` (weight 4.0); both remain `PRACTICAL_APPROXIMATION`.

Prepared root: `artifacts/totaltext-curved-word-regions-refined-v2/`.
`qualification.json` pins the source and refined manifests; the separate
`qualification_native128.json` audits word connectivity at training resolution.
The image bytes are hard-linked from v1 wherever the filesystem permits, so
preparing this target does not duplicate the RGB dataset.
The 20 largest relative foreground contractions can be inspected in
`totaltext-train-refinement-worst20.zip`, with RGB, official pixel GT, v1,
refined-v2 and a v1-minus-v2 panel per TRAIN image. Exact masks are lossless
PNGs; the combined JPEG panels are visual aids.

For manual GPU training on PAPER, run only after checking both qualification
reports:

```bash
bash scripts/train_totaltext_curved_word_regions_host.sh refined
bash scripts/train_totaltext_curved_word_regions_host.sh refined-fusion
```

Neither command modifies the v1 target, network architecture, paper-faithful
configuration, or source dataset. Compare trained checkpoints on held-out data
before drawing conclusions about localization or OCR.

## Filled adaptive word boxes: v3

The separate `adaptive-word-boxes-v3` TRAIN target addresses a visual defect
of refined-v2: the v2 distance envelope can resemble thickened letters. V3
keeps each word a **filled** bounding region. Within each frozen v1 word owner,
it divides the long axis into at most 12 broad strips, uses the official
TRAIN pixel mask to place the top and bottom of each strip, and fills every
pixel between those boundaries. Adjacent strips form a many-corner word box;
pixel GT is not dilated. Short or near-square words, words with no pixel seed,
and unsafe thin/disconnected candidates remain v1. Every source pixel-GT seed
previously inside v1 stays inside v3. V3 can only remove pixels from the
existing disjoint owner, so its words cannot overlap. A native-128 topology
guard also restores a word if downsampling would worsen connectivity.

These strip dimensions and guards are `ASSUMPTION` / `PRACTICAL_APPROXIMATION`,
not paper-specified target construction. No OCR or TEST data were used.
The paper specifies contour-based localization of a binary mask, not this
new TRAIN target. V1, v2 and the paper-faithful localization remain unchanged.

Full TRAIN qualification: 1255 images / 9285 care words; 5999 word boxes
changed before native topology restoration (10 restorations). V3 retains all
39,942,201 source pixel-GT foreground pixels already covered by v1. Filled
foreground area is 102,688,694 pixels versus 104,843,104 for v1 (97.95%
retained) and 86,617,229 for v2 (82.62% of v1). At native 128×128, v1 has
26 absent / 9219 single / 40 multi-component words; v3 has 23 / 9226 / 36.
No per-word connectivity count worsened. These are **TRAIN target diagnostics**,
not segmentation or recognition accuracy; held-out inference is still needed.

Artifacts are isolated in `artifacts/totaltext-adaptive-word-boxes-v3/`:
`qualification.json`, `qualification_native128.json`, the pinned TRAIN
manifest, and `totaltext-train-filled-boxes-v1-v2-v3-comparison.zip`. The ZIP
contains one composite per selected TRAIN image: RGB, original pixel mask,
v1, v2 and v3. The config is
`configs/approximate/segmentation_total_text_adaptive_word_boxes_v3.yaml`.
The user rejected v3 because its strip edges can look stepped rather than
like the intended smooth curved box. Its generated targets and runner are
retained for provenance only; **do not launch a new v3 training run**.

The isolated v3 training configuration was set to **500 epochs**, not the
author-clarified 100. Its protocol and output directory are separately named
`practical_totaltext_adaptive_word_boxes_500_v1` and
`totaltext-adaptive-word-boxes-v3-500ep`; the previous interrupted 100-epoch
run is preserved. No active v3 GPU training process was observed when the
curved-box work began. This 500-epoch config must not be mistaken for the
author-clarified protocol.

No further v3 training is recommended. Better TRAIN target topology alone is
not evidence of better recognition.

## Filled curved boxes: v4 (replacement proposal for rejected v3)

V4 addresses the user's actual request: retain a **filled word box**, but
replace v1's straight polygon sides with continuous curved boundaries. For
each Total-Text curved TRAIN annotation, its two long sides define a curved
centerline. Shape-preserving cubic interpolation and a low-frequency envelope
of the official pixel GT position the upper/lower boundary. The region
between them is completely filled. If an original text pixel needs extra
room, only a *uniform buffer of the entire polygon* is allowed; no glyph
contour is thickened individually. Different words are still separated by
the v1 ownership/contact rules. Source-text coverage or native-128 topology
failures revert the affected annotation, or the entire image, to v1.

This is `ASSUMPTION` / `PRACTICAL_APPROXIMATION`, not a paper-specified
segmentation target. The original v1, v2, rejected v3, and paper localization
implementations are unchanged. It uses TRAIN images/masks only, no OCR or
TEST data. The target construction parameters were fixed before the full
TRAIN audit; no model has yet been trained on v4.

TRAIN target audit: 1255 images / 9285 care words, 5170 curved annotations,
4016 word regions changed relative to v1. The foreground area is
113,066,884 pixels versus v1's 104,843,104; every one of the 39,942,201
source pixel-GT seeds already covered by v1 is retained. There are 232
geometric per-word fallbacks, 346 source/native safeguard fallbacks, and 184
whole-image fallbacks. These are target-shape counts, not accuracy. The
prepared target and qualification live under
`artifacts/totaltext-curved-word-boxes-v4/`; the comparison ZIP presents one
composite image per selected TRAIN sample (RGB, pixel GT, v1, rejected v3,
curved v4). Inspect the images before authorizing any GPU training; a smooth
box can look more natural but still hurt downstream localization or OCR.
At native 128×128 the word-component distribution is unchanged from v1:
26 absent, 9219 single and 40 multi-component; there are zero cross-word
8-neighbour contacts and zero worsened word topologies. The native audit is
`qualification_native128.json`.

The visual overlay ZIP `train-original-gt-v4-overlays-best25.zip` contains 26
single-image composites: 25 TRAIN cases with prominent curved-target changes,
no v4 fallback, and at least 98% original pixel-GT coverage, plus sample 147.
Panels show original RGB, original pixel GT, v4, and a pixel-exact overlay
(white=both, red=GT only, cyan=v4 only). These are **selected visual examples**,
not the best model predictions; no v4 model has been trained yet.

The isolated `segmentation_total_text_curved_word_boxes_v4.yaml` configuration
trains v4 for 500 epochs, as separately requested by the user. This departs
from the clarified 100-epoch paper protocol. Run on PAPER only, in foreground:

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str && bash scripts/run_totaltext_curved_box_v4_train_and_test_host.sh
```

The host runner verifies the v4 target manifest and both qualifications,
trains without touching v1/v3 runs, then evaluates the selected checkpoint on
all 300 official Total-Text TEST images using the existing native-128
FindContours + PARSeq diagnostic. The TEST output reports one-to-one matched
care words, case-sensitive and casefold exact counts, geometry, proposal count,
and per-image masks/crops/OCR. It is a `PRACTICAL_APPROXIMATION`, not a
paper Table 5 result. A partially interrupted training run is not overwritten
automatically; the user must resume it explicitly after inspecting its state.

## v1 checkpoint: direct FindContours recognition diagnostic

The completed standard v1 run selected `epoch-0092.pt` from training foreground
IoU. To test the simple contour baseline on all 300 official Total-Text TEST
images, the host runner applies `sigmoid >= 0.5` to the checkpoint's native
128×128 output, calls OpenCV `findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`,
filters unpadded native bounding-box area at 150, pads by 6 native pixels, keeps
at most 10 boxes by padded area, remaps them to original RGB, and recognizes
the crops with the qualified PARSeq runtime. GT is used only after crop
generation for word-level one-to-one IoU>=0.5 evaluation. This contour-area
interpretation is a repository-inspired `PRACTICAL_APPROXIMATION`; it is not a
Table 5 claim.

Run on PAPER from any directory:

```bash
bash /extra1/bmartelli/projects/icdar2025-context-driven-str/scripts/run_totaltext_v1_findcontours_parseq_host.sh /extra1/bmartelli
```

It saves resumable masks/crops/OCR shards and `results.json` under
`/extra1/bmartelli/context-str-runs/totaltext-v1-findcontours-parseq-test-v1/`.
