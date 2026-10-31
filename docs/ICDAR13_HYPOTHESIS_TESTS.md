# ICDAR2013: vocabulary, crop, cross-fitting and mixed-data controls

**Approximate reproduction / research extension.** This user-authorized study
tests the proposed causes of weak U-Net/PARSeq F1 and image-router ranking.
The local paper sections3.2,3.5 and4.1 and the existing matched-training,
vocabulary and router implementations were inspected before changes.

Use the existing RGB-disjoint183 FIT /23 selection /23 CAL /233 TEST cohorts,
all1095 TEST care words, unchanged casefold/alphanumeric lexical micro F1,
native256 care/ignore and declared no-alphanumeric light emissions. Historical
TEST exposure remains a limitation. This is not strict paper reproduction or
official polygon Hmean. No GT is changed, no new dependency is installed and
all DeepSolo predictions are authenticated cached real outputs.

1. Compare no vocabulary, released IC13 Generic keep-unmatched, and a fixed
   conservative Generic policy on the identical emitted PARSeq words/crops.
   Select on23 selection images only. Frozen original gate decisions and
   GT-oracle61-image controls distinguish correction from routing changes.
2. Read fixed GT-identity crops with the same PARSeq at padding0 and4. Diagnose
   missing viable crops, viable wrong readings and correct rejected readings
   at declared bboxIoU. This privileged spatial diagnostic neither changes the
   lexical metric nor supplies predictions/features to a deployable router.
3. Train five frozen-recipe U-Nets from the exact qualified Total-Text source
   for30 epochs, each excluding its FIT fold completely. No fold GT selects
   checkpoints, crop policies or weights. Compare the same20 small gate
   candidates on seen versus OOF FIT observations. A separate five-head control
   adds eight pre-filter descriptors and allows observable scoring of empty
   outputs. Selection chooses each head; exact TEST budget is61/233 (26.18%).
4. Compare existing IC-only training with a new matched Total-Text replay arm.
   Keep every IC fitting-image exposure, sample order, optimizer step, model,
   initialization and Adam settings. Each IC batch adds a same-size TT batch
   with loss0.5 IC +0.5 TT, accumulated sequentially in FP32. Source epoch0 and
   epochs10/20/30 are eligible on selection only. Additional source examples
   and domain weighting are the explicit intervention; training compute rises.

Every fold starts before IC fine-tuning, not from the final IC checkpoint.
OOF models fit146/147 images versus final183; fold-size shift is unresolved.
The23-image selection set is reused and contains only64 care words. Cross-fitting
addresses expert-seen FIT optimism, not all sources of selection uncertainty.
SynthText pretraining remains a conditional later intervention; this first
controlled mixed-data experiment tests Total-Text replay directly.

Before mixed training, a secondary fixed30-epoch control was explicitly declared
in `configs/approximate/icdar13_hypothesis_fixed_epoch_control_v1.json`.
It distinguishes the equal-training-duration comparison from the selected-epoch
pipeline comparison. It never chooses weights from TEST; if selection chooses
epoch30, exact state equality permits reusing the already-scored outputs.
Otherwise evaluate the saved last checkpoint separately, retaining both outcomes.

Source: `scripts/run_icdar13_hypothesis_tests.py`,
`recognition/icdar13_hypothesis_tests.py`,
`configs/approximate/icdar13_hypothesis_tests_v1.json`.
External root:
`/extra1/bmartelli/context-str-runs/icdar13-hypothesis-tests-20261029-v1`.
Smoke and full results are separated and preserve resolved configuration,
source snapshot, raw readings, maps, weights, per-image counts and failures.
The completed run and independent audit are linked below.

Classification: PAPER_SPECIFIED architecture/fallback motivation only;
REPOSITORY_SPECIFIED qualified components/targets and losses; ASSUMPTION folds,
word metric, vocabulary/GT-crop diagnostics, gate grids, mixed objective and
selection; HARDWARE_DEVIATION actual RTX3090, sequential domain gradients and
model/cache execution; UNRESOLVED generalization, upstream overlap and strict
artifact/metric equivalence. See `ICDAR13-HYPOTHESIS-001`.

## Qualification and frozen full-data diagnostics

Eighteen focused tests pass. The complete smoke_v2 executes two one-epoch
folds, physical8 sequential IC/TT backward, GT crop reading, vocabulary replay
and45 fresh small-head fits. Its four-image results are smoke tests, not quality
estimates. The mixed backward peaks at8,947,160,576 allocated and12,062,818,304
reserved bytes. No OOM occurred. Every1130 no-augmentation TT donor target passes
its actual768/384 kernel loader. Router and mixed-data claims below come only
from the completed full run and its independent audit.

Full frozen vocabulary controls, all233 TEST images and1095 care words:

| PARSeq word policy | Selection F1 % | CAL F1 % | TEST F1 % | TEST TP / FP / FN | DeepSolo images/total (%) |
| --- | ---: | ---: | ---: | --- | --- |
| Original |90.3226|72.7273|73.8756|772 /223 /323|0/233 (0.00%)|
| Generic keep-unmatched |83.8710|69.5187|74.2584|776 /219 /319|0/233 (0.00%)|
| Conservative Generic |90.3226|71.6578|74.0670|774 /221 /321|0/233 (0.00%)|

The selection-only winner is original PARSeq (prefer no change on an exact
tie). Generic's small TEST gain is a descriptive control, not authorization to
promote it after observing TEST. Keep-unmatched changes87 normalized readings,
helps10 images and harms6, for net4TP. Surface case/punctuation-only edits do not
count toward those87. For example, correct strings may be moved to the wrong
nearby dictionary entry; the conservative policy can change Future to FUTURES
or &WINES to SWINES. Generic assistance alone does not explain most of this
light-expert gap. Direct word readings and edits remain auditable.

Privileged fixed-identity GT crops are read correctly1057/1095 (96.5297%) tight,
and1063/1095 (97.0776%) with original-pixel padding4. These are identity accuracy
diagnostics with DeepSolo0/233 (0.00%), not deployable lexical F1 or absolute
ceilings. At bboxIoU>=.5,293 GT words have no viable raw proposed crop;271 of
those are readable from their tight GT crop. Another54 have a viable crop but
no correct raw reading,41 readable from tight GT;45 have a correct viable raw
reading rejected by the native confidence filter,42 readable from tight GT.
The remaining703 have a correct emitted viable crop. This spatial partition is
distinct from primary lexical multiset matching, which does not require IoU.
It must not be substituted for the323 lexical FN or attributed wholly to the
segmentation mask without a separate crop-assembly ablation.

Preserved preliminary failures: an accidental unused configuration import,
historical TT TRAIN40 checksum mismatch before training, and a duplicate
logging keyword after the first one-epoch smoke update. The TT drift (also149,
336) was already documented elsewhere; the existing contained checksum-verified
view restores original manifest-identical bytes without changing labels or
relaxing integrity. smoke_v2 corrects only the logging defect and records frozen
encoder verification. Failed smoke and setup records remain in the external root.
The first full independent report stopped before writing its report because its
CSV recount lacked two fields required by aggregation. Those fields were added,
a regression test was run, and the full independent audit passed. The corrected
first report remains preserved; `report_v2` also checks the older frozen gate.

## Completed out-of-fold light expert evidence

Five fixed30-epoch folds cover every183 FIT image exactly once, with each image
excluded from the corresponding U-Net fitting set. At the unchanged PARSeq/crop
policy, the OOF light expert has546 TP,99 FP,144 FN on690 care words:
**81.7978% F1, DeepSolo0/183 (0.00%)**. The original seen-image light outputs
on those same183 images had628 TP,31 FP,62 FN and93.1060% F1,
DeepSolo0/183 (0.00%). Generic DeepSolo on the same FIT cohort had569 TP,
128 FP,121 FN and82.0476% F1, DeepSolo183/183 (100.00%).

This confirms a large expert-seen fitting advantage in the router's original
training inputs. It does not by itself explain the entire TEST gap: OOF models
fit146/147 images and FIT has a different word population from TEST.

## Completed trained-router results

The full run completed five30-epoch folds and30 mixed-data epochs. It trained
20 seen-feature heads,20 OOF-feature heads and five OOF/raw/no-guard heads from
scratch. The independent report verified the45 fitted heads, all183 fold
exclusions, the frozen ResNet encoder states, original IC training image order,
the official CSV counts and exact61-image ranking. Selected head and crop policy
were fixed before TEST evaluation. The same saved, authenticated DeepSolo outputs
were used for all logical routes; **zero new DeepSolo inference calls** occurred.

| Router or expert, full TEST | TP / FP / FN | Lexical F1 % | DeepSolo images/total (%) |
| --- | --- | ---: | --- |
| Original light expert |772 /223 /323|73.8756|0/233 (0.00%)|
| Older frozen trained gate, same current experts |831 /136 /264|80.6014|61/233 (26.18%)|
| Previous matched trained router; seen FIT observations |832 /185 /263|78.7879|61/233 (26.18%)|
| **New trained router; OOF FIT observations** |851 /156 /244|**80.9705**|61/233 (26.18%)|
| OOF + raw descriptors + empty-output scoring |846 /158 /249|80.6098|61/233 (26.18%)|
| GT-perfect subset for the original experts, diagnostic |925 /76 /170|88.2634|61/233 (26.18%)|
| DeepSolo alone, cached real predictions |977 /95 /118|90.1707|233/233 (100.00%)|

The OOF router gains **2.1826 percentage points** over the matched seen-data
router and **0.3691 points** over the older frozen gate, both at exactly61/233
DeepSolo decisions on the same expert outputs. Its selected head uses OCR and map
features only. Its frozen score has Spearman0.265 with observed image benefit,
versus -0.052 for the seen-data head. Of61 chosen fallbacks,43 have positive
local benefit, six negative and12 zero; the seen-data gate has33/7/21.
The new gate still accepts76 images where the DeepSolo fallback has positive
local benefit, leaving a **7.2929-point** gap to the original-expert GT oracle.
This oracle uses TEST labels and is not deployable. The selection-only head
winner survives16/23 leave-one-selection-image-out replays; three winners occur.

The paired2000-image bootstrap for OOF minus previous router is
**[-0.598,5.123] F1 percentage points** at nominal95%. This interval
conditions on fixed trained models and does not include head-selection
uncertainty, upstream historical TEST exposure or training-set variation.
Against the older frozen gate the corresponding interval is **[-1.598,2.449]**
points. The point improvements are useful evidence, but both intervals include zero.
Raw descriptors plus removal of the empty-output guard are not supported here:
their TEST F1 is0.3607 points below the OOF control and their gate sends11
locally harmful images to DeepSolo versus six. Because feature additions and
guard removal were bundled, this test does not attribute the difference to
either one individually.

## Completed mixed Total-Text replay result

All30 mixed epochs preserve the original IC sample order and23 optimizer steps
per epoch. Each uses183 IC and183 TT images; an independent replay verifies
all30 TT order hashes, and all1130 available TT donor images appear across
training. The intervention adds donor data, domain weighting and compute.
The frozen encoder, 768-square input, FP32, physical batch8 and original Adam
settings are unchanged. Peak mixed training memory was9.212 GB allocated and
12.065 GB reserved on the RTX3090; no OOM occurred. This hardware differs from
the paper's environment and is recorded as a hardware deviation.

| Light expert checkpoint | Selection F1 % | CAL F1 % | TEST TP / FP / FN | TEST F1 % | DeepSolo images/total (%) |
| --- | ---: | ---: | --- | ---: | --- |
| Original IC-only selected |90.3226|72.7273|772 /223 /323|73.8756|0/233 (0.00%)|
| Mixed selected on23 selection images, epoch10 |81.8898|75.6477|793 /253 /302|74.0775|0/233 (0.00%)|
| Mixed fixed epoch30, predeclared secondary control |81.2500|78.3069|787 /230 /308|74.5265|0/233 (0.00%)|

The selection-chosen mixed checkpoint improves TEST by only0.2019 points;
fixed epoch30 improves by0.6509. Their paired nominal95% intervals against
the original light expert are respectively **[-2.042,2.470]** and
**[-1.434,2.766]** points. Epoch30 performs better on TEST and CAL despite
lower selection F1, evidence that23-image selection is noisy; TEST did not
choose either checkpoint. The fixed epoch result is a secondary analysis, not
a retroactively selected replacement.

The mixed selected model's spatial diagnostic has344 GT words without a viable
crop (versus293 originally),63 viable but misread (versus54), and29 correct
raw readings rejected by the confidence filter (versus45). It gains21 lexical
TP but also30 FP. These are different views of the output and must not be
subtracted as though spatial matches were the lexical micro-F1 accounting.
With the old frozen router and61/233 (26.18%) logical DeepSolo decisions, the
mixed expert yields **78.1801%**, below the original route's78.7879%.
Thus this matched replay does not solve the localization/crop weakness and does
not improve the complete system at the fixed old gate. Its GT-perfect61-image
diagnostic is88.1324%, also slightly below the original experts'88.2634%.

Vocabulary-only correction was likewise not selected: Generic keep-unmatched
gains0.3828 TEST points, but loses6.4516 selection points and its paired
interval [-0.500,1.290] includes zero. The conservative policy ties the
original on selection and gains0.1914 TEST points; the declared tie break
retains the original. Removing the confidence filter outright increases TP
from772 to828 but FP from223 to773, lowering F1 to61.4243% with DeepSolo
0/233 (0.00%). Fixed-GT PARSeq readability is high, while missing viable
crops dominate the privileged spatial failure partition. The next focused
localization experiment should separate mask recall, connected-component/crop
assembly, and confidence admission on a new selection population before any
threshold or checkpoint is promoted.

These results remain **approximate reproduction**, not Table3 segmentation F1,
official polygon Hmean or strict paper Table5 accuracy. Historical upstream
TEST exposure and the small reused selection set limit generalization claims.
Cached routing gives exact logical fallback counts, not online latency. The
complete machine-readable comparison, per-sample outputs and plot are under
`/extra1/bmartelli/context-str-runs/icdar13-hypothesis-tests-20261029-v1/full/report_v2`;
the lightweight completion receipt is
`provenance/ICDAR13_HYPOTHESIS_TESTS_COMPLETION_V1.json`.
