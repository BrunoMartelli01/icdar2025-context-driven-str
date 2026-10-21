# Restart and failed-run storage cleanup

The failed SynthText v1 run occupies about601MiB. Its `last.pt` and `best.pt`
have epoch0/global_step0 and tensor names/shapes/dtypes/values identical to the
preserved qualified `unet-word-research-v1/control/best.pt`. The original baseline
is required and is never a cleanup candidate. Failed-run contracts, diagnostics,
target qualification and logs are retained.

Before cleanup, the original163,008,932-byte SynthText manifest was archived to
`artifacts/diagnostics/unet-restart-storage-check/failed-synthtext-manifest.json.gz`.
Its decompressed SHA256 is
`59e244b23db1409cecec0e861d23ea18f5fb1d004663d276c4188cc6037eb002`, equal to the
original. The14,992,954-byte archive preserves all input annotations and provenance.
Removing the two epoch-zero copies and uncompressed duplicate manifest reclaims
about586MiB net. Dataset images, gt.mat, download receipts and prior trained
checkpoints are excluded.

The cleanup helper defaults to read-only. `--apply` checks the failed phase,
epoch history, source weights, checksum backup and runner locks before recording
an audit and deleting its explicit three-file inventory. Failed checks stop the
operation. It is tested for trained/different/different-dtype weights, nonzero
epochs/steps, changed backups, published runs, symlinks and successful repeats.

On2026-10-13 `/extra1` had only about3.7GiB free, below the launcher's12GB reserve.
`/var/tmp` is on the separate NVMe filesystem with about25GiB free. Its ordinary
30-day tmpfiles entry is currently commented; this remains temporary storage,
so archive final results. Changing the output path preserves model/data/training
settings and is recorded by the existing run contract. Repeating the training
command resumes that same new output root.

Run on `bmartelli@paper.dinfo.unifi.it`:

```bash
set -e
cd /extra1/bmartelli/projects/icdar2025-context-driven-str
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venvs/context-str-main-py311/bin/python \
  -m scripts.cleanup_failed_unet_detection_run \
  --failed-run /extra1/bmartelli/context-str-runs/unet-detection-methods-synthtext-small-v1 \
  --manifest-backup artifacts/diagnostics/unet-restart-storage-check/failed-synthtext-manifest.json.gz \
  --apply

bash scripts/run_unet_detection_methods_host.sh \
  artifacts/totaltext-curved-word-boxes-v4 \
  /extra1/bmartelli/context-str-runs/unet-word-research-v1 \
  /var/tmp/bmartelli-context-str-runs/unet-detection-methods-synthtext-small-v2 \
  /extra1/bmartelli/context-str-artifacts/parseq \
  --synthtext-root /extra1/bmartelli/datasets/context-str/SynthText \
  --config configs/approximate/unet_detection_methods_synthtext_targets_v2.json
```

No dataset download occurs. This environment can only read the external run
directory, so the actual cleanup/training are performed by the host command.
Engineering verification:21 cleanup/pruner tests passed, AST and diff checks
passed; the real failed-run dry-run passed without removing external files.
Machine-readable audit and optional completed-run plan are in
`artifacts/diagnostics/unet-restart-storage-check/`.

The separately reviewed optional plan would remove498 intermediate checkpoints
of the SUCCESS/full500-epoch separated-word run, recovering229.07GiB while
preserving the selected epoch485, last epoch500, all logs/reports and datasets.
This concerns a successful run and requires separate user approval; it is not
included in the default command above. After that approval only, adding
`--completed-plan artifacts/diagnostics/unet-restart-storage-check/completed-run-pruning-plan.json`
to the cleanup command uses the existing pruning implementation with repeated
manifest, selected/last SHA and sidecar checks. Those intermediate weights cannot
be recovered without an external backup. No completed-run pruning has been applied.
