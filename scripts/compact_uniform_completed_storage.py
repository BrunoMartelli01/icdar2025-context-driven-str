"""Explicitly compact completed, unselected optimizer states without losing models."""

import argparse
import os
from pathlib import Path

import torch

from scripts.compact_completed_uniform_optimizers import equal
from scripts.run_uniform_unet_training import read, write
from segmentation.uniform_unet_fusion import file_hash


def compact(folders, receipt):
    if receipt.exists():
        raise ValueError('Preserve the existing compaction receipt')
    if len(set(folders)) != len(folders):
        raise ValueError('Duplicate checkpoint directory')
    # Fail before any mutation if an experiment is active on an affected root.
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        if proc.parent.name == str(os.getpid()):
            continue
        try:
            args = proc.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b'--run-root' in args and any(str(p).encode() in args for p in folders):
            raise ValueError('A process is using an affected experiment')
    for folder in folders:
        complete = read(folder/'complete.json')
        if not (complete['status'] == 'COMPLETE' and complete['selected_epoch'] < complete['epochs']):
            raise ValueError('Only a completed nonselected final optimizer may be compacted')
        if read(folder/'independent_training_audit.json')['status'] != 'PASS':
            raise ValueError('Independent training audit required')
        if file_hash(folder/'best.pt') != complete['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed')
        if (folder/'last.compaction_tmp.pt').exists():
            raise ValueError('Preserve previous temporary checkpoint')
    records = []
    for folder in folders:
        path = folder/'last.pt'
        old = torch.load(path, map_location='cpu', weights_only=False)
        if 'optimizer_state_dict' not in old:
            raise ValueError('Optimizer state already absent')
        new = {k: v for k, v in old.items() if k != 'optimizer_state_dict'}
        temp = folder/'last.compaction_tmp.pt'
        torch.save(new, temp)
        loaded = torch.load(temp, map_location='cpu', weights_only=False)
        if not equal(new, loaded):
            raise ValueError('Nonoptimizer state changed during serialization')
        record = dict(path=str(path), old_sha256=file_hash(path), new_sha256=file_hash(temp),
            old_bytes=path.stat().st_size, new_bytes=temp.stat().st_size,
            best_sha256=file_hash(folder/'best.pt'), plan_sha256=file_hash(folder/'resolved_configuration.json'),
            model_tensors=len(loaded['model_state_dict']), all_nonoptimizer_state_exact=True,
            optimizer_continuation_available=False)
        write(receipt, dict(status='IN_PROGRESS', completed=records, pending=record))
        temp.replace(path)
        records.append(record)
        del old, new, loaded
    write(receipt, dict(status='COMPLETE', cases=records,
        reclaimed_bytes=sum(r['old_bytes']-r['new_bytes'] for r in records),
        source_sha256=file_hash(__file__), scope='Selected weights untouched; every other final checkpoint field roundtrip-equal. Exact optimizer continuation removed from completed nonselected endpoints.'))
    print('Compacted', len(records), 'optimizer states; reclaimed GiB', sum(r['old_bytes']-r['new_bytes'] for r in records)/2**30)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run-root', type=Path, action='append', required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    compact(a.run_root, a.receipt)
