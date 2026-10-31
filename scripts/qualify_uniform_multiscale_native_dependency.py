"""Link a cached blend to its original genuine native-neural qualification."""

import argparse
from pathlib import Path

from scripts.qualify_uniform_scaled_frontend import inputs
from scripts.run_uniform_unet_training import read, write
from segmentation.uniform_unet_fusion import file_hash


def qualify(path):
    c,u,source,plan,records,ids=inputs(path);root=Path(c['output_root'])
    parent=Path(c['scaled_export']).parent
    audit_path=parent/'native_parity_audit.json';probe_path=parent/'native_probe/complete.json'
    audit=read(audit_path);probe=read(probe_path);complete=read(root/'scaled/complete.json')
    if not (audit['status']=='PASS' and audit['maps_and_context_exact'] and audit['raw_readings_and_crop_hashes_exact']):
        raise ValueError('Original actual-neural native qualification must pass')
    if (probe['checkpoint_sha256']!=complete['checkpoint_sha256'] or
        probe['checkpoint_sha256']!=read(source/'complete.json')['checkpoint_sha256'] or
        probe['crop_settings']!=complete['crop_settings'] or probe['ground_truth_used'] or
        probe['actual_unet_image_inferences']!=audit['images']):
        raise ValueError('Native qualification belongs to another frontend')
    evidence=read(root/'source_map_evidence.json')
    for p in (audit_path,probe_path):evidence[str(p)]=file_hash(p)
    write(root/'source_map_evidence.json',evidence)
    write(root/'native_parity_audit.json',dict(status='PASS',kind='REUSED_AUTHENTICATED_NATIVE_QUALIFICATION',
        original_audit=str(audit_path),original_audit_sha256=file_hash(audit_path),
        original_probe=str(probe_path),original_probe_sha256=file_hash(probe_path),
        original_native_neural_images=audit['images'],actual_new_neural_calls=0,
        scope='Existing genuine native768 map/context/transcript/crop parity, same checkpoint and crop policy. No new neural parity or blend-quality result is claimed.',
        source_sha256=file_hash(__file__)))
    print('Authenticated native-neural dependency linked; new neural calls0.')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    qualify(p.parse_args().config)
