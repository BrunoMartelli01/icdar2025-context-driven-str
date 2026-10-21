# context-str-main snapshot

This directory records the environment actually solved and validated on
2026-08-18. The manager is `uv 0.12.5`; the canonical named environment lives at
`/home/bruno/projects/icdar2025-context-driven-str/.venvs/context-str-main` and uses
the persistent uv-managed CPython 3.11.11 installation under `.venvs/python/`.
The repo-local location was selected as an equivalent persistent user-owned directory
because it is writable, remains with the project, and can be excluded atomically from
Git via `.gitignore`. The uv executable is preserved at `.venvs/uv/uv`.

- `environment.yml` records direct pins, exclusions, and PARSeq provenance.
- `requirements.in` records the direct non-PyTorch install request.
- `installed.txt` is the exact sorted `uv pip freeze` result after solving.
- `environment_metadata.json` records platform/runtime provenance.
- `validation.json` is the successful physical RTX 3070 validation report.

`installed.txt` is an exact-version snapshot, not a hash-complete lock. A uv
virtual environment created through `uv pip` does not provide an installed-state
lock export; generating `uv.lock` from a newly solved project could differ from
the environment that was actually tested. Recreate from the snapshot with both
PyPI and the official PyTorch CUDA 12.4 index, then run `uv pip check` and
`scripts/check_main_environment.py` before treating the environment as validated.

The persistent environment was recreated without dependency resolution: the already
validated installed package tree was copied byte-for-byte and only absolute
console-script shebangs were mechanically relocated. Its sorted `uv pip freeze`
output exactly matches `installed.txt`, and `uv pip check` reports all 70 packages
compatible. Direct post-relocation CUDA validation was attempted, but the current
no-escalation execution sandbox cannot access NVML/CUDA. CPU PARSeq strict loading and
inference passed; the earlier physical-host CUDA report remains historical rather than
being presented as a post-relocation GPU rerun.

The official PARSeq source is not vendored or installed from its old package
metadata. Supply either a clean checkout at revision
`1902db043c029a7e03a3818c616c06600af574be`, or the extracted GitHub commit archive
whose SHA256 is
`f57289cb1fc7fe41f488a52ea48f8a7f114252ef405b0b64efc1d528c29b48ef`, to the
validation script. Archive mode verifies every extracted file against the frozen
tarball. The checkpoint remains the separate v1.0.0 `parseq-bb5792a6.pt` artifact
with SHA256
`bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00`.
