"""Engineering-only sequential GPU lifecycle tracing for the 8 GB target host."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class SequentialLifecycle:
    events: List[Dict[str, Any]] = field(default_factory=list)

    def stage_completed(self, stage: str, latency_seconds: float, placement: str) -> None:
        self.events.append({
            "event": "stage_completed",
            "stage": stage,
            "latency_seconds": float(latency_seconds),
            "placement": placement,
        })

    def release(self, stage: str, component: Any) -> None:
        """Release references, then perform auditable allocator housekeeping."""
        import torch

        release = getattr(component, "release", None)
        if not callable(release):
            raise RuntimeError(
                "Sequential lifecycle requires {} to expose an explicit release()".format(stage)
            )
        before = None
        if torch.cuda.is_available():
            index = int(torch.cuda.current_device())
            torch.cuda.synchronize(index)
            before = {
                "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "reserved_bytes": int(torch.cuda.memory_reserved(index)),
            }
        started = time.perf_counter()
        details = release()
        gc.collect()
        after = None
        if torch.cuda.is_available():
            index = int(torch.cuda.current_device())
            torch.cuda.empty_cache()
            torch.cuda.synchronize(index)
            after = {
                "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "reserved_bytes": int(torch.cuda.memory_reserved(index)),
            }
        self.events.append({
            "event": "component_released",
            "stage": stage,
            "latency_seconds": time.perf_counter() - started,
            "release_details": dict(details or {}),
            "cuda_memory_before": before,
            "cuda_memory_after": after,
            "engineering_action": (
                "release Python/model references; gc.collect; torch.cuda.empty_cache when CUDA visible"
            ),
            "scientific_algorithm_step": False,
        })
