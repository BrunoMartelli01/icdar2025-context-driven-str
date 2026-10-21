# Guarded word GT V3 — approximate reproduction

This implements the three user-requested safeguards from NATIVE128-FAIL-001:
check connectivity after the real transform, retreat from excessive erosion of
fragile words, and verify separation together with crop coverage. It is a new
GT experiment, not a paper-specified shrinking recipe or a strict reproduction.
Paper sections3.2–3.3 do not prescribe the following target repair.

## Runtime construction

Use the pinned V2 compact source images, masks, owner maps, validity and
separators. Replay the existing paired flip/rotation/resize without modifying
the historical transform. The V1 owner map follows exactly the same seeded
geometry and is nearest-resized to the actual128 grid as a reference envelope.

For each source word, after the complete native target/separator/validity
processing:

1. Keep the V2 core when it is one8-connected component and satisfies the
   configured area, both-axis bbox-span, crop-coverage and thin-word guards.
2. Otherwise restore progressively more source pixels: grow the core inside
   the SAME V1 owner and valid region, using eight linearly spaced native
   radii0.25..2. If no candidate passes, try the full permitted reference owner.
3. Exclude every pixel adjacent to another retained word in its 8-neighbourhood.
   A successful update never erases existing compact pixels or changes another
   word. Recompute the negative separator so restored foreground is not
   simultaneously supervised as background.
4. If no candidate passes ALL guards, retain every original V2 fragment and
   record the failed conditions. This means the current policy did not find a
   valid repair; it does not prove that no alternative global repair exists.

Words are visited in ascending source-ID order. Previously accepted repairs
remain fixed; there is no claim of globally optimal packing/order independence.
Every source word remains in the denominator, including words cropped out by
rotation or absent after rasterization. Secondary islands are never discarded
to manufacture a connected result.

## Explicit initial assumptions

The resolved recipe is `configs/approximate/guarded_word_gt_v3.json`:

- Max growth2 and 8 steps reuse the previous native inset budget and backoff
  count. The linear native-growth schedule is a new `ASSUMPTION`.
- Minimum area20% and span80% reuse V2 bounds, with span now checked on BOTH
  axes. Their suitability has not been independently selected.
- Coverage padding2 comes from the existing approximate localization config,
  not a numeric paper requirement or new TEST optimization. Coverage concerns
  the transformed valid V1 reference bbox, not proof of perfect character crops
  from future network predictions.
- Reference word bbox short side<=2 native pixels protects the full source
  shape when clearance permits. This is an initial untuned thin-word rule.
- Restoration may use nearest-owner support locally instead of the old
  bilinear-threshold target. This explicit `ASSUMPTION` changes supervision,
  not RGB interpolation. It is not silently inserted into older runs.

Classification, alternatives and unresolved issues are recorded under
WORD-GT-003 in `docs/PAPER_AMBIGUITIES.md`. V1/V2 files, official TEST labels,
legacy transforms and strict/default configurations remain unchanged.

## Training interface and per-sample evidence

Select V3 explicitly:

```python
from torch.utils.data import DataLoader
from segmentation.guarded_word_dataset import make_audited_guarded_dataset

dataset = make_audited_guarded_dataset(
    "configs/approximate/guarded_word_gt_v3.json",
    augmentation_enabled=True,
)
dataset.transform.set_epoch(0)
loader = DataLoader(dataset, batch_size=8, num_workers=2,
                    shuffle=False, persistent_workers=False)
batch = next(iter(loader))
assert batch["mask"].shape[1:] == (1, 128, 128)
assert batch["target_variant"][0] == "guarded-word-targets-v3"
```

The wrapper delegates actual target construction unchanged to the qualified
adapter. Each sample carries `native_owners`, `target_variant`, the recipe hash,
collatable `gt_guard_stats`, and `gt_guard_audit_json` with every source word and
failure reason. These survive worker boundaries. Save/aggregate this evidence
in the future training run; do not rely on a parent-process `last_audit` when
using workers. For validation use `augmentation_enabled=False` and the returned
V3 owners/expected IDs when computing target-based Word-F1.

Set the epoch before creating each iterator. `persistent_workers=False` follows
the qualification; persistent workers need explicit epoch-state propagation
and are not qualified here. Retain original full-manifest sample indices when
using a1130/125 Subset, so RNG geometry is identical across target variants.

The manifest still describes the pinned V2 SOURCE files. The resolved V3 recipe
and transform metadata describe the final training targets. Historical training
scripts continue using their historical GT unless a dedicated V3 experiment
explicitly selects this factory and records the V3 recipe. No training run was
started as part of target construction.

## Qualification and native previews

The CPU qualification uses all 1255 official TRAIN photographs and 9285 source
word IDs, without TEST input. It compares before/after under no augmentation
and the actual deterministic flip/rotation of epochs0 and 1. OpenCV independently
checks the SciPy component decisions and verifies that no connected component
contains more than one owner. Same-epoch/index replays check determinism.

```bash
.venvs/context-str-main-py311/bin/python -u -m scripts.qualify_guarded_word_gt \
  --config configs/approximate/guarded_word_gt_v3.json \
  --output artifacts/diagnostics/guarded-word-gt-v3-NEW/full
```

Add `--limit-images 3` in a separate output directory for the real smoke test.
The worker-boundary smoke is independently reproducible with:

```bash
.venvs/context-str-main-py311/bin/python -m scripts.qualify_guarded_word_dataloader \
  --config configs/approximate/guarded_word_gt_v3.json \
  --output artifacts/diagnostics/guarded-word-gt-v3-NEW/worker_smoke.json \
  --epoch 1 --num-workers 2 --batch-size 2 \
  --sample-ids total_text:1006 total_text:114 total_text:1009
```

`full/native_preview/` contains unaugmented native 128 owner/mask PNGs and hashes
for inspection. They are NOT source-space training masks: feeding them to the
old loader would rotate/resize already-native targets again. Use the runtime
factory for training. Source photos, OCR crops and checkpoints are not copied.

Results, unresolved-word lists and a measured V1/V2/V3 comparison are isolated
in `artifacts/diagnostics/guarded-word-gt-v3-20261017/`. A controlled training and
validation experiment is still required to measure OCR improvement. Two sampled
augmentation epochs do not establish behavior for every future random view.

## Completed qualification,2026-10-17

1255 images and 9285 original words per view. All words remain in denominators, including rotation-clipped words or words absent on grid128. Before/after counts use the same deterministic transform:

| View | Fragmented V2 | Fragmented V3 | Missing V2 | Missing V3 | Words with unresolved constraints | Inter-word contacts V3 |
|---|---:|---:|---:|---:|---:|---:|
| Without augmentation | 275 | 23 | 8 | 5 | 240 | 0 |
| Augmentation epoch 0 | 300 | 28 | 30 | 19 | 267 | 0 |
| Augmentation epoch 1 | 278 | 22 | 25 | 18 | 248 | 0 |

42 focused tests, the real3-image smoke and 3765 independent OpenCV image checks pass. CPU batch 2/2-worker DataLoader output matches direct indexing and the full epoch 1 hashes. Complete resolved outputs: `artifacts/diagnostics/guarded-word-gt-v3-20261017/`. No training or OCR improvement is claimed. Unresolved per-word conditions remain recorded and retained, including31 unaugmented coverage failures.
