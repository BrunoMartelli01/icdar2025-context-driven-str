import os
import shutil

import pytest

from scripts.deduplicate_completed_generator_copies import run
from scripts.run_uniform_unet_training import write, read
from segmentation.uniform_unet_fusion import file_hash


def test_identical_bytes_and_metadata_are_preserved_with_no_duplicate_storage(tmp_path):
    source=tmp_path/'historical';source.mkdir();(source/'checkpoint.pt').write_bytes(b'qualified model and metadata'*1000)
    root=tmp_path/'completed';target=root/'generators/historical_0';target.mkdir(parents=True)
    shutil.copy2(source/'checkpoint.pt',target/'checkpoint.pt');digest=file_hash(source/'checkpoint.pt')
    write(root/'completion.json',dict(status='SUCCESS'));write(root/'independent_audit.json',dict(status='PASS'))
    reuse=dict(status='PASS',stages=[dict(source=str(source),stage='historical_0',regenerated_records_exact=True,
        files_sha256={'checkpoint.pt':'invalid'})]);write(root/'generator_reuse.json',reuse)
    before=(target/'checkpoint.pt').stat().st_ino;receipt=tmp_path/'receipt.json'
    with pytest.raises(ValueError,match='hash-identical'):run(root,receipt)
    assert not receipt.exists() and (target/'checkpoint.pt').stat().st_ino==before
    reuse['stages'][0]['files_sha256']['checkpoint.pt']=digest;write(root/'generator_reuse.json',reuse);run(root,receipt)
    assert os.path.samefile(source/'checkpoint.pt',target/'checkpoint.pt')
    assert file_hash(target/'checkpoint.pt')==digest and read(receipt)['status']=='COMPLETE'
