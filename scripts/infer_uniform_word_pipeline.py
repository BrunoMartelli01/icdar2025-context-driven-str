"""Run the shared approximate word pipeline from an RGB-only JSON manifest."""

import argparse
from pathlib import Path
import time
import traceback

from pipeline.uniform_word_pipeline import run
from scripts.run_uniform_unet_training import write


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--dataset', choices=['icdar13', 'icdar2015', 'totaltext'], required=True)
    p.add_argument('--policy-quota', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output = a.output.resolve()
    if a.output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError('Neural outputs must be stored outside the source repository')
    try:
        run(a.config, a.manifest, a.dataset, a.policy_quota, a.output)
    except BaseException as exc:
        write(a.output/('failure_'+str(time.time_ns())+'.json'), dict(status='FAILED', error=repr(exc),
            traceback=traceback.format_exc(), out_of_memory='out of memory' in str(exc).lower()))
        raise
