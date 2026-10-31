"""Hardlink byte-identical completed historical copies, preserving every path/hash."""

import argparse
import os
from pathlib import Path

from scripts.run_uniform_unet_training import read, write
from segmentation.uniform_unet_fusion import file_hash


def run(root, receipt):
    if receipt.exists():raise ValueError('Preserve the existing storage receipt')
    if read(root/'completion.json')['status']!='SUCCESS' or read(root/'independent_audit.json')['status']!='PASS':
        raise ValueError('Completed independently audited destination required')
    reuse=read(root/'generator_reuse.json')
    if reuse['status']!='PASS':raise ValueError('Qualified copy-reuse evidence required')
    jobs=[]
    for row in reuse['stages']:
        if not row['regenerated_records_exact']:raise ValueError('Historical records changed')
        source=Path(row['source'])/'checkpoint.pt';target=root/'generators'/row['stage']/'checkpoint.pt'
        a,b=source.stat(),target.stat();digest=row['files_sha256']['checkpoint.pt']
        if (a.st_dev!=b.st_dev or a.st_ino==b.st_ino or
            (a.st_uid,a.st_gid,a.st_mode,a.st_mtime_ns)!=(b.st_uid,b.st_gid,b.st_mode,b.st_mtime_ns)):
            raise ValueError('Distinct same-filesystem copies with identical metadata required')
        if file_hash(source)!=digest or file_hash(target)!=digest:raise ValueError('Copies are not hash-identical')
        temporary=target.with_name('checkpoint.deduplication_tmp.pt')
        if temporary.exists():raise ValueError('Preserve existing temporary')
        jobs.append((source,target,temporary,digest,b))
    rows=[]
    for source,target,temporary,digest,old_stat in jobs:
        if target.stat().st_ino!=old_stat.st_ino:raise ValueError('Copy changed after qualification')
        os.link(source,temporary)
        write(receipt,dict(status='IN_PROGRESS',completed=rows,pending=dict(source=str(source),target=str(target),sha256=digest)))
        temporary.replace(target)
        if source.stat().st_ino!=target.stat().st_ino or file_hash(target)!=digest:raise ValueError('Linked bytes changed')
        rows.append(dict(source=str(source),target=str(target),sha256=digest,
            original_copy_inode=old_stat.st_ino,shared_inode=target.stat().st_ino,
            reclaimed_allocated_bytes=old_stat.st_blocks*512,preserved_bytes=old_stat.st_size,
            model_optimizer_metadata_and_paths_preserved=True))
    write(receipt,dict(status='COMPLETE',cases=rows,reclaimed_allocated_bytes=sum(r['reclaimed_allocated_bytes'] for r in rows),
        source_sha256=file_hash(__file__),reuse_evidence_sha256=file_hash(root/'generator_reuse.json'),
        scope='Completed immutable historical copies only. All checkpoint bytes, paths, permissions and mtime preserved; future replacement writes must be atomic, never in-place.'))
    print('Identical completed copies linked:',len(rows),'reclaimed GiB',sum(r['reclaimed_allocated_bytes'] for r in rows)/2**30,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();run(a.root,a.receipt)
