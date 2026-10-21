#!/usr/bin/env python3
"""Path-verifying bootstrap for the staged, repository-owned M6 JSON worker."""

import importlib
import os
import sys
from pathlib import Path


def _require_under(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            "M6 JSON worker bootstrap resolved {} outside staged bundle root: {}"
            .format(label, resolved)
        ) from error
    return resolved


def _load_staged_worker():
    entrypoint = Path(__file__).resolve()
    bundle_root = entrypoint.parent.resolve()
    expected_package_root = (bundle_root / "deepsolo").resolve()
    sys.path.insert(0, str(bundle_root))
    for name in list(sys.modules):
        if name == "deepsolo" or name.startswith("deepsolo."):
            del sys.modules[name]
    importlib.invalidate_caches()
    package = importlib.import_module("deepsolo")
    package_file = getattr(package, "__file__", None)
    if not package_file:
        raise RuntimeError("M6 JSON worker bootstrap resolved deepsolo without __file__")
    resolved_package = _require_under(
        Path(package_file), expected_package_root, "deepsolo package"
    )
    worker = importlib.import_module("deepsolo.worker")
    worker_file = getattr(worker, "__file__", None)
    if not worker_file:
        raise RuntimeError("M6 JSON worker bootstrap resolved deepsolo.worker without __file__")
    resolved_worker = _require_under(
        Path(worker_file), expected_package_root, "deepsolo.worker"
    )
    os.environ["CONTEXT_STR_M6_WORKER_ENTRYPOINT_PATH"] = str(entrypoint)
    os.environ["CONTEXT_STR_M6_WORKER_PACKAGE_PATH"] = str(resolved_package)
    os.environ["CONTEXT_STR_M6_WORKER_MODULE_PATH"] = str(resolved_worker)
    return worker


if __name__ == "__main__":
    raise SystemExit(_load_staged_worker().main())
