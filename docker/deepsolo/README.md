# DeepSolo M6 Docker runtime

This directory reconstructs the operational boundary around the locally qualified
`APPROXIMATE` DeepSolo runtime. Read
[`docs/DEEPSOLO_RUNTIME.md`](../../docs/DEEPSOLO_RUNTIME.md) before changing or
running it.

The authoritative qualified object is the local image
`context-str-deepsolo:m6-qualified-20260818` with image ID
`sha256:30e0e99d23279060e72b22fe3911a2e42b72de55e70be6349fb06ec37b35755c`.
The exact base-image construction history was not captured. Consequently, the
Dockerfile is a validation/reconstruction wrapper around an explicitly supplied
legacy base image; a successful rebuild receives a different identity and is not
automatically qualified.

Copy `.env.example` to the ignored `.env`, set absolute source/checkpoint paths,
and keep the source at commit `dbadae995035246bad3376c7a44c015c69e9b313`.
The checkpoint directory must contain
`tt_res50_finetune_synth-tt-mlt-13-15.pth` with SHA256
`ac2a01b3d8df0ee42e4632ff57a0d3a20a5706fb0ac1c5ae33b2455d0d90da68`.

Commands:

```bash
scripts/deepsolo_start.sh
scripts/deepsolo_smoke.sh
```

`deepsolo_start.sh` reuses the authoritative persistent container when present.
If it must create the container, it bind-mounts source and checkpoints instead of
embedding either in image layers. `deepsolo_build.sh` is only a reconstruction aid:
it validates a supplied legacy base and invokes the frozen upstream editable build
against the persistent source. It never qualifies the derived image by itself.

For the existing qualified snapshot, `DEEPSOLO_CHECKPOINT_DIR` is recovery metadata
only; the checkpoint already exists at `/opt/checkpoints`. The host adapter verifies
`DEEPSOLO_SOURCE_ROOT` through the host checkout and passes an invocation-scoped
source-identity record to the worker, so neither Git nor `.git` is required in the
container.

Normal `python -m pytest` uses mocks and never runs Docker or CUDA.
