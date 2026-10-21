# PAPER host and commands

User-designated destination: **`bmartelli@paper.dinfo.unfi.it`**.
The spelling is preserved exactly. Some historical documents contain
`paper.dinfo.unifi.it`; this discrepancy has not been resolved by silently
changing the destination. Verify DNS/connectivity when SSH is needed.

The host check during this refactor returned `hostname -f = paper` and
`whoami = bmartelli`: commands in this session already run on PAPER.
No second SSH connection to the same machine is needed.

## Before running commands

```bash
hostname -f
whoami
pwd
```

Repository on PAPER:
`/extra1/bmartelli/projects/icdar2025-context-driven-str`.
When already on PAPER:

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str
python3 -m research groups
```

From another machine, for a read-only remote command:

```bash
ssh bmartelli@paper.dinfo.unfi.it \
  'cd /extra1/bmartelli/projects/icdar2025-context-driven-str && python3 -m research groups'
```

If the designated name does not resolve, report the error and verify the
destination; do not change aliases, SSH keys, or network settings to bypass it.

## Operational locations

| Content | Historical location on PAPER |
| --- | --- |
| Repository | `/extra1/bmartelli/projects/icdar2025-context-driven-str` |
| Persistent runs | `/extra1/bmartelli/context-str-runs` |
| External weights and artifacts | `/extra1/bmartelli/context-str-artifacts` |
| Datasets | `/extra1/bmartelli/datasets/context-str` |
| Existing main Python | `.venvs/context-str-main-py311/bin/python` inside the repository |
| Existing YOLO Python | `.venvs/context-str-yolo-py311/bin/python` inside the repository |
| Existing legacy DeepSolo | `.venvs/context-str-deepsolo-host/` inside the repository |

These are operational references, not defaults to embed in scientific models or
configs. Some runs use `/var/tmp` or `/tmp`; verify their presence rather than
assuming it.

For reports without models, use `python3` without activating a heavy environment:

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str
export CONTEXT_STR_RUNS_ROOT=/extra1/bmartelli/context-str-runs
export CONTEXT_STR_ARTIFACTS_ROOT=/extra1/bmartelli/context-str-artifacts
python3 -m research check
python3 -m research compare yolo-test-equal
```

For inference/training, select the environment documented for that component.
Before working on DeepSolo, read [DEEPSOLO_RUNTIME.md](DEEPSOLO_RUNTIME.md),
the M6 manifest, and the DeepSolo entries in STRICT status. Do not merge the
environments or install packages merely to inspect reports.

Before a GPU run, inspect `nvidia-smi` and active processes. RTX 3070
qualifications, RTX 3090 runs, and the paper's environment remain distinct
hardware evidence. Record the actual device and deviations in the run.
