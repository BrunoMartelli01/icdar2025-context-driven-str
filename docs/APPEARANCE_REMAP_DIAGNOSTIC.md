# Appearance-guided local re-map v1

Status: implemented prototype, evaluated once on the full frozen TRAIN set.
**Not calibrated or promoted.** Result class `PRACTICAL_APPROXIMATION`; scientific mode
`approximate`; claim label `approximate reproduction`; Table 5 accuracy stays null.

This is a separate CPU localizer and TRAIN-only diagnostic entry point. It does
not replace Algorithm1, practical-remap-v1/v2, or the existing local rescue. It is
not automatically connected to contextual routing. The segmentation checkpoint,
native128 threshold, PARSeq, BLIP2, MPNet and DeepSolo are untouched.

## Evidence and scope

The local paper `papers/2503.15639v2.pdf`, section 3.3 / Algorithm1, specifies
components, area filtering, padded original-image crops, area ordering and top10.
It does not specify an appearance-based word grouping stage. That stage and all
its new thresholds are `ASSUMPTION` (L-APP-001), not newly resolved paper details.
The numeric area150/pad6 settings are `REPOSITORY_SPECIFIED`; applying the area
filter to grouped foreground is an explicit approximation.

The motivation is the observed trade-off of geometry-only grouping: recovering
fragments often also joins neighbouring words. Additional RGB appearance is a
new source of evidence, not a guarantee of word boundaries. Stroke/appearance
grouping has a classical precedent in Epshtein et al.,
[Detecting Text in Natural Scenes with Stroke Width Transform](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/201020CVPR20TextDetection.pdf).
This code is **not** an implementation of that full detector or its learned
parameter values. Its foreground-thickness estimate is only a mask-based proxy.

## Implemented algorithm

1. Validate uint8 binary native128. Project its labels with exact integer nearest
   sampling to the supplied RGB image. Do not interpolate probabilities or infer
   segmentation. Source dimensions must be positive. Check the actual one-to-one
   native/original component identity, including for source dimensions below128;
   reject any actual component loss, split or merge, not the image size itself.
2. Call the existing, unchanged v1 localizer in original coordinates. Keep all
   pre-cap base groups immutable, including their true foreground pixel areas.
3. Describe each raw component using median foreground CIELAB color, foreground
   minus background median L, and a foreground-thickness proxy. Background comes
   from a bbox neighbourhood excluding **all** predicted foreground. The proxy
   is twice the median exact L2 distance-transform radius at 3x3 local maxima,
   with explicit zero padding at ROI borders. Mask holes remain background.
   These measurements use original-image pixel geometry, not square128 ratios.
4. Select immediate left/right neighbours using direct vertical compatibility,
   without transitive global line formation. Select neighbours before inspecting
   appearance: a rejected neighbour cannot be skipped to reach a farther one.
   Require mutual neighbourhood, normalized gap, and gap/internal-spacing gates.
   Internal spacing uses positive gaps between disjoint original X intervals;
   singleton/no-positive-gap bases have no relative-spacing gate.
5. Test pairs, and base+left+right triples when both adjacent edges pass. Every
   raw descriptor in the entire hypothesis must satisfy the appearance gates:
   maximum pairwise DeltaE76, thickness ratio, common contrast polarity and
   contrast-magnitude ratio. Missing foreground/background/contrast evidence
   forbids a **new merge**, but never discards its v1 base group.
6. Resolve competing hypotheses deterministically by smaller maximum color
   difference, smaller thickness ratio, smaller edge gap, more base groups (at
   most3), then base IDs. Consume each base at most once. Newly merged groups
   never become search nodes. Thus no chain can grow past3 immutable base groups,
   though each base may already contain multiple raw components from v1.
7. Union original unpadded boxes and sum disjoint **original foreground** areas.
   Never count filled-contour interiors or artificial bridging pixels. Apply
   area150, pad6, the existing area/spatial ordering and top10. Return crops from
   the unmodified RGB plus raw IDs, features, acceptance/rejection reasons.

The numeric appearance settings in
`configs/approximate/localization_appearance_guided_local_remap_v1.yaml` are
unvalidated engineering starting values declared before any experiment. No new
grid, GT-based choice, model fit or parameter selection has been performed.

## Controlled comparison

The standalone diagnostic reports three outputs from exactly one cached mask:

| Output | Grouping | Final area filter |
|---|---|---|
| `v1_frozen` | Existing frozen v1 | Existing v1 behaviour: no Amin filter |
| `v1_matched_postprocessing` | Same immutable v1 groups | area150, identical to new mode |
| `appearance_remap` | New bounded local merges | area150 |

This makes area-filter effects visible separately from new grouping. With new
merges disabled, the last two outputs must be identical in boxes, area, order,
and crop bytes. With min_area_original=0 and merges disabled, output must also
match frozen v1. These are synthetic implementation tests, not alternative
calibration variants.

Geometry uses the existing diagnostic greedy one-to-one IoU matching to keep the
baseline comparable. Per-GT best-IoU counts and per-crop wordness have separate
denominators. Wordness keeps the declared .25 GT coverage significance criterion
and MULTIWORD priority. They are diagnostic boxes, not Table5 recognition
accuracy or definitive causal under/overmerge classifications. No OCR runs.

## Run once on CPU, foreground

From the repository root (paths are explicit inputs, not embedded in code):

```bash
PY=.venvs/context-str-main-py311/bin/python
C=/tmp/icdar13-train-native-masks-epoch0098/manifest.json
D=/extra1/bmartelli/datasets/context-str/icdar13
V=/tmp/practical-remap-train-calibration-v1.json
CFG=configs/approximate/localization_appearance_guided_local_remap_v1.yaml
OUT=/tmp/appearance-guided-local-remap-train-v1
CUDA_VISIBLE_DEVICES='' "$PY" -u -m scripts.diagnose_appearance_remap_train \
  --config "$CFG" \
  --cache-manifest "$C" \
  --train-manifest manifests/icdar13_segmentation_author_protocol.yaml \
  --data-root "$D" \
  --v1-calibration "$V" \
  --output "$OUT.json" \
  --csv-output "$OUT.csv"
```

All229 PNG hashes and decoded masks, source images, GT BMP and TXT identities
are checked against the pinned cache/calibration. The original v1 is recomputed
from the same cached masks and must reproduce its frozen full-TRAIN geometry
before a report is accepted. The manifest checksum proves the cache's checkpoint
identity; no checkpoint/model is loaded. Existing result files cannot be
overwritten. JSON/CSV include hashes, configuration and per-image/word evidence.
The script has no device switch, segmentation fallback, OCR or TEST evaluator.

## Implementation checks

The targeted CPU suite passed 153 tests covering localization, dataset handling,
author-protocol segmentation metrics/preprocessing and the frozen adapter. It
includes 29 new appearance-remap/configuration/provenance tests. Python compilation
and `git diff --check` passed.

A smoke test used only TRAIN sample100 RGB and its cached native mask, after
verifying the pinned cache/calibration manifests and sample hashes. It confirmed
native-component identity, unchanged input arrays and JSON-serializable trace
output. No GT, segmentation inference, CUDA, OCR, training or parameter selection
was used for this smoke test. These checks establish implementation behaviour,
not recognition or localization performance.

## Frozen TRAIN execution

The first attempt stopped before publishing any report at sample245 (102x422):
an unnecessary source-size guard rejected it despite all33 native components
surviving the frozen nearest projection one-to-one. Only this guard was corrected;
actual topology changes remain errors. No masks, grouping parameters or baseline
code changed. Three regression cases cover short images and true CC loss/merging.

The complete rerun accounts for all229 images /849 words, with no CUDA,
segmentation inference, OCR, TEST access, grid or training. It reproduced the
frozen-v1 geometry exactly. JSON/CSV population, code/configuration hashes and
per-word aggregate counts were independently checked after publication.

| Metric | Frozen v1 | v1 + matched area filter | Appearance remap |
|---|---:|---:|---:|
| GT best IoU>=.25 |679|671|671|
| GT best IoU>=.50 |568|567|566|
| GT best IoU>=.75 |324|324|333|
| Proposal precision@.50 |.472153|.582734|.592050|
| Recall@.50 |.669022|.667845|.666667|
| Proposal F1@.50 |.553606|.622393|.627147|
| Retained proposals |1203|973|956|
| WORD_GOOD |552|551|550|
| WORD_PARTIAL |130|113|109|
| MULTIWORD_MERGE |37|37|40|
| UNMATCHED |484|272|257|

Of43 accepted merge hypotheses before filtering/capping, the final proposals
recover3 words at IoU>=.50 but lose4 relative to the matched control (3 recovered,
5 lost relative to frozen v1). Appearance grouping's F1 increment over the matched
control is only .004754; most of the increment over frozen v1 is due to the area
filter. Word recovery/WORD_GOOD do not increase, and MULTIWORD rises. **No promotion
is justified for the intended word-recovery objective.** No thresholds were changed
after inspecting these results. IoU>=.75 improves, but is not evidence of OCR gains.

- JSON: `/tmp/appearance-guided-local-remap-train-v1.json`
  SHA256 `770ce5b13a7ddcb09c84985e1dbc9ed49e70190676d633bf248035714032df06`
- CSV: `/tmp/appearance-guided-local-remap-train-v1.csv`
  SHA256 `5a83eb5a6241ada7115e698564e18db4c609ca720a0a1078a90a9277c08f2fc7`

## Limits

- Same-colour neighbouring words can still be merged; these cues do not reveal
  word semantics. Inconsistent colours within a legitimate word can veto rescue.
- Native128 may erase text or quantize thickness. Projecting it cannot restore
  missing detail. Low-evidence components remain as baseline groups, subject to
  the separately reported final area/top10 filters.
- Baseline v1 overmerges are not split by this method. Whole-group gates only
  prevent some **new** mistakes; this is not a global repair of v1.
- The single controlled TRAIN diagnostic above does not establish improved word
  recovery or OCR performance. No TEST/OCR results may select thresholds.
