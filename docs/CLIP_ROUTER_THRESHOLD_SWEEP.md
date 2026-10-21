# Frozen CLIP router threshold curves

This is **approximate reproduction**, a descriptive experimental extension of
paper section3.5/Algorithm2. Table6's paper confidence ablation is MPNet/fuzzy
`alpha/beta/tau`; this CLIP threshold sweep does not reproduce that table.

Vary only the complete-image CLIP acceptance threshold: keep all PARSeq outputs
of an accepted image, or replace them all with DeepSolo outputs of its original
full image. The fixed DeepSolo instance-score threshold remains .4. CLIP B16,
Text+Kernel epoch61, PARSeq transcripts, TRAIN80 mid-CDF references, crop recipes,
matching/ignore rules and all previous predictions remain frozen. GT labels are
used only to report the performance curve. No new threshold is selected using
TEST labels. STRICT configurations are unchanged.

TRAIN is the already analysed125 held-out TRAIN images/983 care GT words. TEST
contains the separate official300 images/2204 care GT words. Never aggregate
these two populations together. The new source TRAIN result now exists: at the
previous threshold .146906 it has664 TP/212 FP/319 FN,876 predicted care words,
75.7991% detected-word precision and55.5649% TN0 accuracy.

Report two different word counts: `words_retained_parseq` counts care words
kept by the router before fallback; `combined_model.detected_words` counts all
final care predictions, including DeepSolo. Raw and ignored output counts are
preserved separately. Precision is `TP/(TP+FP)` and TN0 accuracy is
`TP/(TP+FP+FN)`. Casefold-exact one-to-one bbox IoU>=.5 and original evaluator
polygon/EXIF/native-support ignore are unchanged. Counts can increase at an
individual threshold: replacing an image can add more DeepSolo detections than
the number of PARSeq words removed. Missing DeepSolo outputs prevent a complete
point and leave the final metric null; they never become empty outputs or omitted
images. The125/300 and983/2204 denominators stay fixed.

## TEST300

| Threshold | PARSeq images | Retained PARSeq words | Final words | TP | FP | FN | Precision | TN0 accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|300|2509|2509|1215|1294|989|48.43%|34.73%|
|.025|264|2084|2363|1272|1091|932|53.83%|38.60%|
|.05|227|1734|2237|1333|904|871|59.59%|42.89%|
|.075|202|1459|2190|1390|800|814|63.47%|46.27%|
|.10|178|1249|2133|1411|722|793|66.15%|48.22%|
|.125|160|1057|2089|1439|650|765|68.88%|50.42%|
|.146906|149|978|2069|1459|610|745|70.52%|51.85%|
|.175|139|927|2064|1461|603|743|70.78%|52.05%|
|.20|135|896|2060|1466|594|738|71.17%|52.39%|
|.25|120|732|2046|1495|551|709|73.07%|54.26%|
|.30|104|580|2016|1507|509|697|74.75%|55.55%|
|.40|74|353|2010|1548|462|656|77.01%|58.06%|
|.50|41|137|2002|1572|430|632|78.52%|59.68%|
|.6192174372838609|23|63|2005|1579|426|625|78.75%|60.04%|
|.70|14|39|2008|1584|424|620|78.88%|60.27%|
|.80|5|8|2008|1588|420|616|79.08%|60.52%|
|1|0|0|2008|1588|420|616|79.08%|60.52%|

The .5 point uses DeepSolo on259/300 images; its TN0 accuracy is .8372
percentage points below DeepSolo on all300. The .146906 point uses151/300,
losing8.6704 points. These are descriptive comparisons, not a TEST-selected
deployment threshold. Invocation count is measured; wall-time/FLOP savings are
not established. The TEST curve can be fully computed from the already audited
300 DeepSolo outputs, without new inference.

## TRAIN125

| Threshold | PARSeq images | Retained PARSeq words | Final words | TP | FP | FN | Precision | TN0 accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|124|1029|1029|480|549|503|46.65%|31.33%|
|.025|113|893|980|512|468|471|52.24%|35.29%|
|.05|96|691|948|564|384|419|59.49%|41.26%|
|.075|91|639|923|574|349|409|62.19%|43.09%|
|.10|76|471|903|625|278|358|69.21%|49.56%|
|.125|69|409|891|643|248|340|72.17%|52.23%|
|.146906|62|343|876|664|212|319|75.80%|55.56%|

The one empty PARSeq image always routes to DeepSolo, including at threshold0.
At higher thresholds, respectively7/10/16/24/37/49/55/60/62/62 further TRAIN
fallback outputs are missing for .175/.2/.25/.3/.4/.5/.619217/.7/.8/1. All
PARSeq retained-word counts remain available, but final metrics remain null.
Completing all125 DeepSolo outputs needs62 additional original-image inferences.

## Run and inspect

Isolated configurations:
`configs/approximate/clip_router_threshold_sweep_v1.json` and
`configs/approximate/clip_router_threshold_sweep_local_paths_v1.json` pin the
source plan, both split results and independent TEST audit hashes.

```bash
# Read-only input checks.
.venvs/context-str-main-py311/bin/python -m scripts.sweep_clip_router_e2e --mode approximate --check

# Cached-only recombination: complete TEST curve and available TRAIN points.
.venvs/context-str-main-py311/bin/python -m scripts.sweep_clip_router_e2e --mode approximate

# GPU host: execute only the missing62 original-image TRAIN DeepSolo requests,
# resume deterministic caches, then regenerate complete separate curves.
.venvs/context-str-main-py311/bin/python -m scripts.sweep_clip_router_e2e --mode approximate --execute-missing-train
```

Each job saves a fully resolved provenance plan and new fallback responses
outside Git. Completed branch metrics determine an immutable analysis folder;
after additional inference a new analysis is created, preserving the previous
partial analysis. `latest.json` points to the newest analysis. GPU errors,
raw failure responses and memory receipts propagate through the existing
qualified isolated FP32 CUDA adapter. No main model is reloaded. Additional
GPU inference has not been executed in the managed CUDA-inaccessible sandbox.

Current output:
`/tmp/clip-router-threshold-sweep-v1/4c0f991503c0/analyses/5e57cfedf9f2/`:

- `results.json`, SHA256
  `af860357d0b8f6f1faf495006be4cd64f00326ed79513ce0c0282cd3735411e2`.
- `branch_metrics.json`, SHA256
  `87206aa709288ccffea88b694d88e367a2ab0fbe00734017ed401a2777c0a69b`.
- `thresholds.csv`: both splits,17 grid thresholds, all word counts and ratios.
- `threshold_curve.svg` and `threshold_curve.png`: standalone TEST figure.
- `explore.html`: offline split selector and slider over all observed score
  breakpoints, with explicit unavailable-metric indicators.
- `independent_audit.json`; reproducible auditor
  `/tmp/audit_clip_router_threshold_sweep.py`.

Validation:39 focused tests pass, including inclusive boundaries, forced
fallback, non-monotonic word counts, missing-image denominator guards, cache
identity and absent-GPU failure propagation. The full experiment replays the
original .146906 aggregates exactly on both splits. An independent SciPy
matching/source-geometry audit verifies425 branch populations, all310 TEST
and140 TRAIN curve points, and preserves unavailable points as null. The figure
was inspected visually. Matplotlib/Plotly were unavailable in the existing
environments; no packages were installed. Native SVG/offline HTML were generated
directly; installed Inkscape0.92.5 exported the PNG successfully (exit0), while
emitting D-Bus connection messages in the sandbox. An exploratory broad `/tmp`
file search encountered unreadable service directories; source searches were
subsequently confined to the experiment directories. The shell alias `python`
was unavailable; explicit pinned-venv and `/usr/bin/python3` paths were used.

The complete intended diff was reviewed against the initial workspace snapshot;
existing source/model/runtime files and unrelated user changes are preserved.
