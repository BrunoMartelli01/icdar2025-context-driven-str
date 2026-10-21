# Full official TRAIN1255 Gap refit — approximate reproduction

User explicitly requested training on all images,2026-10-20. Initial response
mentioned1130; inspection clarified full official TRAIN1255, all now included.
Former100calibration and125evaluation images are no longer held out. Old524/526
cannot serve as held-out evaluation of this refit. Decision:
ADASpot-GAP-FULLTRAIN-001. Existing code, paper3.2–3.3/Algorithm1 and supplied
AdaSpot proposal4.1–4.2 inspected before changes.

Training retains preceding copied32.6M Gap decoder, frozen58.2MResNet50/Text/
Kernel, original paired augmentation, RGB512/native256, radius3Gap targets,
AdamW1e-4/decay1e-4, loss.5balancedBCE+.5Dice+.25Textprotection, FP32batch4,
seed202503 and10epochs. Fresh copied decoder and zero Gap head; this is not
continuing priorGap weights. Each epoch must contain every officialTRAIN ID
exactly once; save full IDs and hash. Only Gap decoder/head and copied BN
statistics update; source full state/BN stays fixed, exact teacher output
verified and real decoder gradient/head CUDA update smoke-tested.

Prior100image-calibrated epoch1/.9/preserve fixed before this refit. All10epochs
execute, save1/3/5/10/latest optimizer and RNG. TEST compares fixed epoch1 and
final10diagnostic, prior1030fit Gap and frozenM/K on the same images. No TEST
winner selection, threshold search or revision of deployed epoch. Final10 is
diagnostic even if its TEST happens to score higher.

Initial preflight failed before training on exact official RGB overlap:

- TEST578=TRAIN212:49ee079a6f9d9565f6167c4425db511d87d16aaa59f6bc87d4b2573c7bdb3d0c.
- TEST656=TRAIN81:3bdb83fed0aa26e9370e65113ebc3aca02eed5726d0c1e04e8743865f52d926f.

Preserve the finding. ASSUMPTION: retain all1255TRAIN to honor the request, pin
these exact known overlaps and fail on unknown/changed pairs. Report primary
full300TEST/2204care words, plus secondary298hash-disjoint images with explicit
denominator and duplicate contributions. No hidden exclusion and no independent
TEST300 claim. TEST, including298, was historically explored. OriginalM/K may
already have seen these duplicate RGBs.

Freeze raw neural outputs and allarm instance labels before each image's
validity/transcripts. Ignore only wholly invalid native components; mixed-care
components retained. Validity uses original paired letterbox + nearest native
resize. Keep all careGT/misses and empty/duplicatecare predictions in denominator.
Original RGB half-open bboxIoU>=.5 one-to-one maximum-count matching and exact
Unicode casefold retaining punctuation; geometry and OCR reported separately,
P/R/Hmean and paired gains/losses. Not official polygon scoring or Table5accuracy.
No qualified TESTnative word-owner targets: do not fabricate fusion metrics.

Classifications:

- REPOSITORY_SPECIFIED: qualified source/checkpoints, architecture/training
  numerics, GT/preprocessing, original RGBTESTreader and genuine PARSeqAR1.
- ASSUMPTION: full-data refit, fixed prior choices, known-overlap protocol,
  TESTcare/matching rules and explicit final10diagnostic.
- HARDWARE_DEVIATION: actual CUDA-only FP32batch4TRAIN/batch1segmentationTEST/
  batch4PARSeq, sequential unloading, allocated/reservedVRAM/OOM and failures,
  no CPU fallback,8GB or latency qualification.
- UNRESOLVED: independent transfer/seeds, overfitting, sourcearchitecture/GT
  approximations and full shared-featureAdaSpot. Moreimages/later epochs do not
  guarantee improvement.

Config: configs/approximate/adaspot_gap_fulltrain_v1.json.
Runner: scripts/train_adaspot_gap_fulltrain.py.
All artifacts external; historical audits and STRICT defaults unchanged.

Classification scope: REPOSITORY_SPECIFIED for unchanged training means reuse
of qualified local settings/artifacts; it does not resolve their original
ASSUMPTION status recorded in ADASpot-GAP-DECODER-001. No numerical setting is
thereby promoted to paper evidence. Exact duplicates are observed in these
checksum-frozen local mirror artifacts; this alone does not establish whether
the upstream canonical distribution contains the same duplication.

Completed at
/extra1/bmartelli/context-str-runs/adaspot-gap-fulltrain-v1-20261020.
All10epochs,1255unique images each,314CUDAupdates each,3140total, no skipped
all-invalid batches. Same original weights/BN digest before/after. Fresh Gap
CUDA gradient/head update and exact teacher output checks pass. Training totaled
563.1seconds on RTX3090,FP32batch4; loss .8023082→.4475815. Peak training
allocated3,762,007,040bytes/reserved4,588,568,576bytes; noOOM.78focused tests pass.

Primary300TEST images /2204care words, including2documented RGBduplicates:

| Arm | RGB localized | Exact localized+read | Proposals | OCR Hmean |
| --- | ---: | ---: | ---: | ---: |
| Frozen M/K control | 1598 | 1293 | 2588 | 53.9649% |
| Previous1030fit Gap1 | 1604 | 1299 | 2607 | 54.0012% |
| Full1255Gap1, preselected | 1612 | 1311 | 2624 | 54.3082% |
| Full1255Gap10, diagnostic | 1634 | 1324 | 2677 | 54.2512% |

Secondary298images /2187care words, excluding exactlyTEST578/656:

| Arm | RGB localized | Exact localized+read | Proposals | OCR Hmean |
| --- | ---: | ---: | ---: | ---: |
| Frozen M/K control | 1582 | 1278 | 2571 | 53.7201% |
| Previous1030fit Gap1 | 1588 | 1284 | 2590 | 53.7576% |
| Full1255Gap1, preselected | 1596 | 1296 | 2607 | 54.0676% |
| Full1255Gap10, diagnostic | 1618 | 1309 | 2660 | 54.0128% |

Bothcohorts: relative to previous1030Gap, full1recovers18/damages6 (+12),
full10recovers41/damages16 (+25). Duplicate image contributions17GT/15exact
are identical in allarms; removing them preserves the gains. Full10 gives more
correct words but more proposals/lower precision; full1has slightly higher
Hmean. Do NOT turn this TEST comparison into a new checkpoint-selection rule:
predeclared1/.9/preserve stays selected,10stays diagnostic. This is exploratory
approximate reproduction; no untouchedTEST, officialpolygon or Table5claim.

All2897uniqueRGBcrops read with the same qualified PARSeqAR1 onGPU. Masks,
labels,crops,OCRreadings/confidence,per-samplecounts and fullIDs kept. OCR peak
allocated404,033,536bytes/reserved1,790,967,808bytes are process-level totals,
including retained training tensor allocations, not isolated model-only memory
or a fresh-process latency qualification. No precision or CPUfallback used.

Independent audit PASS: all1255IDs each of10epochs,3140CUDAupdates,
source/checkpoint/data hashes, fixed prior choice,1200TESTimage-arm and1192
secondarysubset image-arm Hungarian-count cases,1200mask replays and original
RGB crop identities. Result SHA256:
bb1b5423e2df4ae92f9af78204c1b23f0c37627f4209790d597f11f8b3e0e186.
Initial overlap preflight failure is preserved in initial_preflight_overlap.json;
no training/runtime/test/OOM failure or suppressed unit test in the completed run.
