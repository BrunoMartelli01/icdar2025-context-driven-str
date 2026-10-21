# Separated full-word masks from downloaded datasets

This is **approximate reproduction**, an isolated change of supervision targets,
not a reproduction of the paper's segmentation results. The paper section 4.1
and Algorithm 1 use text pixel masks; filled word regions and artificial gaps
are not prescribed. Original images, annotations, masks, models and strict
configuration are preserved. Scientific decisions are recorded as WORD-GT-001
in `docs/PAPER_AMBIGUITIES.md`.

## Sources checked on 2026-10-12

| Downloaded dataset | TRAIN / TEST images | Available local GT | Word target source |
|---|---|---|---|
| ICDAR13 | 229 / 233 | TRAIN character RGB masks and blank-separated TXT groups; TEST Task 1 word boxes | TRAIN convex hull of each care character group's box corners; TEST filled word rectangles |
| ICDAR2015 | 1000 / 500 | Official word quadrilaterals and transcripts | Filled original quadrilaterals |
| Total-Text | 1255 / 300 | Word polygons, character masks, and full text-region masks | Frozen V4 curved full-word owners for TRAIN; original word polygons for TEST |
| CTW1500 | 1000 / 500 | Local text-line GT plus recovered VimTS TEST word GT | TEST: curated VimTS polygons and matching original single-token lines; TRAIN word geometry remains unresolved |

Total-Text **already has** full-region binary PNG masks at
`total_text/downloads/Total-Text-Dataset/groundtruth_textregion/Text_Region_Mask/{Train,Test}`
under the user-supplied download root. All 1555 masks were checked: pixel values
are 0/255 and dimensions match images, including `img61.JPG`. They lack an
instance owner map and do not guarantee separated touching words. The new
targets derive owners from word polygons instead.

Online sources:

- [Total-Text official GT](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Groundtruth/Pixel): character and text-region masks are distinct artifacts.
- [ICDAR2015 official challenge report](https://imlab.jp/publication_data/1469/C02.pdf): word quadrilateral annotation and care/ignore flags.
- [CTW1500 upstream](https://github.com/Yuliang-Liu/Curve-Text-Detector): the local baseline is line-level.
- [VimTS data preparation](https://github.com/Yuliang-Liu/VimTS/tree/929dacbee21f5be22eb2091360933da4d3fba122#data-preparation): distributes `gt_ctw1500_word.zip` for TEST. It was recovered directly from the pinned repository, qualified as500 TXT files with all downloaded TEST IDs, variable-vertex polygons,4915 care records and783 ignored records before curation.
- [Pinned CTW word-GT artifact](https://github.com/Yuliang-Liu/VimTS/blob/929dacbee21f5be22eb2091360933da4d3fba122/evaluation/gt_ctw1500_word.zip): SHA256 `c93f48f99d11eb3d5acc8618ec9b5001a69277ee905184ee7c1c26af33011cbf`; Git blob SHA1 `2c53ce6f0e54ed5856e4252d3f73db65f585e497` independently verified against the downloaded342143 bytes.
- [VimTS CTW word-GT download](https://1drv.ms/u/c/030dd3c816e366d8/EZq2H7NOCf9BsCnl0HOyeFgBNL0gEZDQ6gfW4deJcT3w5g?e=HDkfrf).
- [ArT annotation specification](https://rrc-red.cvc.uab.es/?ch=14&com=tasks): Latin word polygons, Chinese line polygons. ArT is another candidate for CTW word geometry; a CTW ID list is not sufficient proof of image correspondence.

Initial network acquisition failed: the shell could not resolve the ArT CDN,
and the web tool could not retrieve the ArT JSON or VimTS OneDrive archive.
The GitHub connector subsequently supplied the exact base64 word-GT ZIP from
the pinned VimTS commit. No package was installed and no text line was divided
into guessed equal-width boxes. ArT and CTW TRAIN word geometry remain unresolved.

## Generation and verification

Run from the repository root with the existing main environment:

```bash
.venvs/context-str-main-py311/bin/python -m scripts.prepare_separated_word_gt \
  --mode approximate \
  --download-root /extra1/bmartelli/datasets/context-str \
  --prepared-root /extra1/bmartelli/datasets/context-str/scene-segmentation-v1 \
  --output-root artifacts/separated-word-gt-v1/v4-wider-gaps \
  --ignore-policy care_region_precedence \
  --v4-root artifacts/totaltext-curved-word-boxes-v4 \
  --clearance 2 --reference-size 128 128 --max-word-area-loss 0.35 \
  --dataset icdar13 icdar2015 total_text --split train test --zip
```

Paths are explicit inputs. The exporter rejects source-tree output, existing
destination splits, invalid sample counts, source image checksum drift and
lost word identities. `--limit 2` in a separate output directory produces
explicitly labeled smoke-test artifacts.

The V4 implementation, configuration, documentation and example image147
were inspected before modifying the target construction. Its curved full-word
TRAIN owners are used directly with checksum and annotation-identity checks.
The other sources use the existing nearest-exclusive-pixel polygon ownership
algorithm, followed by removal of both sides of 8-neighbour contacts. The
largest connected region per word is retained in this initial polygon step;
removed secondary pixels are recorded. All-ignore images remain present.

The user requested words **more separated than V4**. The new gap operation
retreats a word boundary only where another source word is closer than2 pixels
in a128x128 reference frame (anisotropic distances account for image size).
It keeps filled word bodies and curves. It does not erode every boundary.
Source character/stroke pixels within the starting owner map are protected
for Total-Text and ICDAR13. ICDAR13 uses exact-white RGB background conversion,
never a grayscale foreground threshold. ICDAR2015 has no local stroke GT and
uses the topology/area guards only. These thresholds are untuned `ASSUMPTION`
choices, explicitly required on the command line and saved in each config.

A candidate that removes more than35% of one word's area, erases it, or
fragments protected text restores that word. Unseeded secondary islands may
be removed; discarded pixels and restored IDs are reported. V4's source
owners have18 fragmented words at original resolution.16 contain only
unseeded secondary islands and can be cleaned. Two, in samples228 and1111,
have389 stroke pixels in their secondary regions: those are preserved and
reported as inherited fragmentation rather than deleting annotated letters.
Protected text and restoration can limit the achieved clearance. The parameter
is a requested boundary-clearance operation, not a guaranteed minimum gap for
every pair of words.

For non-V4 sources, `care_region_precedence` excludes ignore geometry only
outside explicit care regions. This is required because broad ICDAR13 ignore
boxes can enclose correctly annotated words (e.g.170). Overlap counts are
saved. V4 validity remains byte-identical to its source. The mixed group257
keeps its care character and processes ignored rows separately. Zero-height
ignore boxes in192 are legitimate line geometry; care boxes must remain
nondegenerate. Grouping/hull construction is not established as equivalent
to official ICDAR13 TRAIN Task1 annotations.

Each image has:

- `masks/<id>.png`: full-word binary foreground, 0/255, original resolution.
- `owners/<id>.png`: uint16 word identities, background 0, care owners 1..N.
- `validity/<id>.png`: ignored geometry has value 0, valid pixels 255.
- `separators/<id>.png`: auxiliary contact/gap map, foreground is zero there.
- `annotations/<id>.json`: source geometry/transcripts, hashes, owners and audit.

Each split includes resolved configuration, a complete machine-readable
manifest, and example previews. Dataset ZIPs contain masks and metadata;
source photos are referenced by path and SHA256, not duplicated in ZIPs.
No artifact should be committed to Git.

Every exported word must survive and have zero8-neighbour contact with another
word. Any inherited V4 fragmentation is reported. PNG roundtrip pixels are
checked; each sample includes before/after ownership audits after nearest
resizing to the reference dimensions. Original-resolution guarantees do not
extend automatically to resized masks. Resizing or
augmentation can merge or erase words; ownership must be transformed and
separation re-applied and audited at any future training resolution.
No OCR performance improvement is inferred from these topology checks.

## CTW1500 TEST word-GT curation

This is a new approximate GT variant, distinct from the original line-level
benchmark and from unmodified VimTS annotations. The source ZIP is immutable.
Visual inspection found split single words (`DOUGL` + `ASTON`, `E` + `313` in
image1001), exact duplicate regions (e.g. `64` in1027), nested partial labels
inside an already complete word (`1862`, `64` inside `1862-64.`), and multiple
nested alternative `A` polygons in1239. An export fails if such geometry makes
a word disappear. Both failed source-curation runs and their failure records
are retained outside the final delivery tree.

The explicit approximate cleanup:

1. Remove exactly identical instances, recording both source row IDs.
2. Remove a lexical fragment only when its complete raster is contained in
   a larger word polygon and its normalized text is a strict substring of that
   larger word. Normalization keeps Unicode letters/numbers and casefolds text;
   original transcriptions are preserved. Unrelated overlaps are not removed.
3. For equal-text nested alternative polygons, keep the tighter source polygon;
   raster-equivalent alternatives retain the earlier row. Record every removal.
4. Match word centers to original care line polygons. Merge parts only when the
   original line has exactly one whitespace-delimited token and the ordered
   normalized parts reconstruct that token exactly. Use the original polygon
   and unmodified transcription. Multi-token lines are never merged by this rule.
5. Apply the same separation and guarded wider-gap operation as the other datasets.

This interpretation of word boundaries is an `ASSUMPTION`. Source transcription
or annotation errors can remain; the artifact is not an independently hand-audited
official word benchmark. No claim of complete linguistic correctness follows from
zero contacts between owner regions. Original source records and every correction
remain auditable, and no image is silently omitted.

```bash
.venvs/context-str-main-py311/bin/python -m scripts.prepare_ctw1500_separated_word_gt \
  --mode approximate \
  --prepared-root /extra1/bmartelli/datasets/context-str/scene-segmentation-v1 \
  --output-root artifacts/separated-word-gt-v1/v4-wider-gaps \
  --archive artifacts/separated-word-gt-v1/sources/gt_ctw1500_word.zip \
  --source-provenance artifacts/separated-word-gt-v1/sources/ctw1500_word_source.json \
  --ignore-policy care_region_precedence --clearance 2 --reference-size 128 128 \
  --max-word-area-loss 0.35 --single-token-policy restore_matching_original_line --zip
```

The archived source contains TEST only. It cannot produce genuine word-level
geometry for the1000 TRAIN images, which remain explicitly unresolved.

## Complete delivery and qualification

The final delivery is `artifacts/separated-word-gt-v1/v4-wider-gaps/`, containing
4017 original-resolution masks. ZIP filenames explicitly label these artifacts
as approximate: `total_text-approximate-word-gt.zip`,
`icdar2015-approximate-word-gt.zip`, `icdar13-approximate-word-gt.zip`, and
`ctw1500-test-approximate-word-gt.zip`. No CTW TRAIN word masks are supplied.

| Dataset / split | Images | Care words | Words with wider-gap pixels removed | Fragmented words |
|---|---:|---:|---:|---:|
| Total-Text TRAIN | 1255 | 9285 | 4977 | 2 inherited from V4 |
| Total-Text TEST | 300 | 2204 | 1168 | 0 |
| ICDAR2015 TRAIN | 1000 | 4448 | 1261 | 0 |
| ICDAR2015 TEST | 500 | 2077 | 653 | 0 |
| ICDAR13 TRAIN | 229 | 849 | 156 | 0 |
| ICDAR13 TEST | 233 | 1095 | 345 | 0 |
| CTW1500 TEST, curated | 500 | 4776 | 2639 | 0 |

All seven complete exports passed an independent OpenCV binary 8-connected
component audit: zero missing curated owner IDs and zero components shared by
different words. The audit also checked output/image/configuration checksums,
dimensions and mask/owner/validity/separator consistency. Total-Text TRAIN is a
subset of its frozen V4 owners and retains all 40038178 source-stroke pixels
inside those owners; foreground is reduced by 2015089 pixels. Supplementary
packaging checks verify all 500 CTW original-line/source-word annotation pairs,
generation code hashes, and every ZIP member against the corresponding file.

There are 95 CTW single-token line merges, one exact duplicate removal, nine
contained lexical fragment removals and three nested equal-text removals.
The denominator 4776 is the curated word variant, not the raw VimTS count 4915.
No model evaluation is performed. New GT/topology/curation tests and existing
word-region tests pass (27 tests); existing dataset parser and ICDAR13 mask
conversion tests also pass (32 tests). All six source-split smoke exports and
the corrected three-image CTW smoke export pass. Early failures remain recorded
outside the final delivery, as described above.

`qualification.json` covers all seven splits; `independent_audit.json` combines
the two complete independent verification runs. `delivery.json` records ZIP
SHA256s and remaining limitations. `reproduction_source/` preserves the exact
generation code whose hashes occur in resolved configurations. Previews include
`total_text/train/previews/overlap_520.png` (image, V4, new mask, word owners).

To rerun the full independent audit:

```bash
.venvs/context-str-main-py311/bin/python -m scripts.verify_separated_word_gt \
  --root artifacts/separated-word-gt-v1/v4-wider-gaps \
  --v4-root artifacts/totaltext-curved-word-boxes-v4 \
  --dataset total_text icdar2015 icdar13 ctw1500 \
  --output /tmp/separated-word-gt-independent-recheck.json
```

## Train the new Total-Text target and evaluate predicted crops with PARSeq

Run from the repository on the physical host:

```bash
bash scripts/run_totaltext_separated_words_train_and_parseq_host.sh --check
bash scripts/run_totaltext_separated_words_train_and_parseq_host.sh
```

The first command performs CPU preflight only. The second trains the existing
V4 AG-U-Net architecture from its pretrained ResNet50 encoder on all 1255 new
TRAIN masks for 500 epochs, selects a checkpoint using TRAIN foreground IoU,
predicts masks on official TEST-300, saves predicted crops and runs the existing
qualified local PARSeq checkpoint on them. TEST masks, transcripts and source
geometry never generate proposals or select a checkpoint. TEST word annotations
are loaded only after predicted proposals/crops have been frozen, for scoring.

This comparison retains V4's architecture, frozen encoder, FP32/batch8,
optimizer, augmentation, 256x256 images, 128x128 masks and separation handling.
The new config is isolated at
`configs/approximate/segmentation_total_text_separated_words_v1.yaml`; protocol
`practical_totaltext_separated_words_500_v1` uses a new artifact identity and
checksum-bound manifest. Existing V4 and strict configuration remain intact.
An adapter creates `training/manifests/total_text_train.yaml` under the GT root,
using paths relative to an explicit common data root. No photographs or target
pixels are rewritten. It verifies every TRAIN input checksum, disjoint TRAIN/
TEST IDs and three transformed examples, including both inherited V4 fragments.
Existing paired transforms reapply owner contact separation after resizing;
original-resolution word-survival guarantees do not automatically extend to
small words at 128x128 or rotated augmentations.

The evaluation reuses the same approximate native128 contour protocol as the
existing V4 test: sigmoid>=0.5, unpadded bbox area>=150, padding6, maximum10 crops,
sort by clipped padded bbox area. This repository bbox-area protocol differs
from paper component-area localization and is not a strict reproduction.
Segmentation and PARSeq are loaded sequentially using the existing evaluator.
The local PARSeq checkpoint is the already qualified approximate artifact,
not a claim that PARSeq-S or the complete context/DeepSolo paper pipeline runs.

The wrapper prints two distinct rates using one-to-one box IoU>=0.50 matching:

- End-to-end word rate: exact Unicode-casefold text matches divided by **all
  2204 care TEST words**. Missing detections remain in the denominator.
- OCR rate with available geometry: exact casefold/spatial matches divided by
  the count of one-to-one geometry matches. Detection failures are reported
  separately rather than counted in this conditional denominator.

Case-sensitive results, proposal precision/recall/F1, all per-image proposals,
native predicted masks and raw OCR outputs are also saved. Source ignore words
are excluded from care GT; this diagnostic does not perform an official
ignore-overlap prediction filter. In particular, the all-GT word rate is a
recall-like diagnostic and does not penalize false proposals in its denominator.
Do not call it official benchmark accuracy or a Table 5 reproduction.

Default external outputs derive from the host root:

```text
context-str-runs/totaltext-separated-words-v1-500ep/total-text-separated-words-v1-500ep/
context-str-runs/totaltext-separated-words-findcontours-parseq-test-v1/results.json
context-str-runs/totaltext-separated-words-findcontours-parseq-test-v1/crops/
```

Override locations with `SEPARATED_WORD_GT_ROOT`, `SEPARATED_WORD_RUN_ROOT`,
`SEPARATED_WORD_RESULTS_ROOT`, or supply a host-root argument. Training output
must be outside the repository. A partial training run must be resumed with an
explicit checkpoint:

```bash
bash scripts/run_totaltext_separated_words_train_and_parseq_host.sh \
  --resume /path/to/the/run/checkpoints/epoch-0050.pt
```

CPU preflight and focused tests are qualified; no real 500-epoch training or
new TEST accuracy was produced while preparing these commands. Training saves
its existing runtime and peak GPU allocated/reserved-memory metadata; physical
GPU success, memory behavior and recognition performance remain to be measured.

Integration qualification passed 46 focused tests, including a one-step CPU
training/checkpoint fixture and rejection of the wrong target manifest. The
first integration attempt using production worker8 hung: PyTorch's tensor
resource-sharing thread raised `PermissionError: [Errno 1] Operation not
permitted` when opening an AF_UNIX socket in the managed sandbox. Both stalled
attempts were explicitly interrupted. The corrected integration fixture uses
an explicit tiny CPU model and a test-only single-process DataLoader; it is
not qualification of the real AG-U-Net, CUDA or worker8 execution. Production
configuration and runtime error propagation remain unchanged. Run the supplied
training command directly in the physical server terminal.
