"""Finalize checksums after the detached graph/occlusion battery terminates.

Read-only with respect to scientific inputs, predictions and metrics. A failed
battery remains failed; no notification, retraining or parameter change occurs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def finalize(root):
    full = root / 'full'
    status = json.loads((full / 'status.json').read_text())
    if status['status'] != 'SUCCESS':
        raise RuntimeError('Scientific battery did not complete successfully: ' + repr(status))
    results = json.loads((full / 'results.json').read_text())
    audit = json.loads((full / 'independent_audit.json').read_text())
    if results['status'] != 'SUCCESS' or audit['status'] != 'PASS':
        raise RuntimeError('Scientific results or independent matching audit failed')
    manifest = {}
    for path in sorted(full.rglob('*')):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
            manifest[str(path.relative_to(full))] = dict(bytes=path.stat().st_size,sha256=digest.hexdigest())
    output = root / 'full_artifact_manifest.json'
    output.write_text(json.dumps(dict(status='PASS',artifacts=manifest,
        image_arm_checks=audit['image_arm_checks'],scientific_results_unchanged=True),indent=2)+'\n')
    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--battery-pid',required=True,type=int)
    parser.add_argument('--poll-seconds',type=float,default=20.)
    args = parser.parse_args()
    if args.poll_seconds <= 0: raise ValueError('Positive polling interval required')
    status_path = args.root / 'full/status.json'
    try:
        while True:
            if status_path.exists():
                state = json.loads(status_path.read_text())['status']
                if state == 'FAILED': raise RuntimeError('Battery failed; preserve failure.json')
                if state == 'SUCCESS': break
            try: os.kill(args.battery_pid,0)
            except ProcessLookupError: raise RuntimeError('Battery exited without a terminal result')
            time.sleep(args.poll_seconds)
        count = finalize(args.root)
        value = dict(status='PASS',artifacts=count)
    except Exception as exc:
        value = dict(status='FAILED',error=repr(exc))
        (args.root/'finalization_status.json').write_text(json.dumps(value,indent=2)+'\n')
        raise
    (args.root/'finalization_status.json').write_text(json.dumps(value,indent=2)+'\n')


if __name__ == '__main__': main()
