"""Bind existing qualified full-FIT validation views to the matching FIT policy."""

import argparse
from pathlib import Path
import shutil

import numpy as np

from scripts.run_uniform_unet_training import read,write
from scripts.build_uniform_multiscale_word_banks import independent_rows
from scripts.icdar_budget_data import key
from recognition.uniform_encoder_context import raw_context
from segmentation.uniform_unet_fusion import checked_mode,file_hash


def run(path):
    c=read(path);checked_mode(path,c);root=Path(c['output_root']);u=read(c['utility_config'])
    policy_sha=file_hash(root/'frontend_policy.json')
    assert policy_sha==read(root/'contract.json')['frontend_policy_sha256']
    for ds in c['dataset_order']:
        profile_path=Path(c['source_validation_profiles'][ds])
        assert file_hash(profile_path)==c['pinned_inputs'][str(profile_path)]
        vp=read(profile_path);candidate=Path(vp['output_root']);source=Path(u['training_root'])/ds
        plan=read(source/'resolved_configuration.json');selected=read(source/'complete.json')
        ids=plan['partitions']['selection'];records={r['sample_id']:r for r in plan['train_records']}
        assert vp['word_augmentation']==c['frontend_policy']['word_augmentation']
        assert read(candidate/'independent_map_audit.json')['status']=='PASS'
        assert read(candidate/'scaled/complete.json')['checkpoint_sha256']==selected['checkpoint_sha256']
        large=Path(vp['scaled_export']);assert read(large/'complete.json')['checkpoint_sha256']==selected['checkpoint_sha256']
        ni=read(source/'export_train/index.json');li=read(large/'index.json');ci=read(candidate/'scaled/index.json')
        assert set(ci)==set(ids) and not set(ids)&set(plan['partitions']['fit'])
        context=raw_context(u,{s:records[s] for s in ids});out=root/'full'/ds/'validation_bank'
        out.mkdir(parents=True,exist_ok=False);index={};evidence={}
        for sid in ids:
            entries=[ni[sid],li[sid],ci[sid]];samples=[]
            for entry in entries:
                assert file_hash(entry['path'])==entry['sha256'];value=read(entry['path'])
                assert not value['ground_truth_used'] and value['image_sha256']==records[sid]['image_sha256']
                evidence[entry['path']]=entry['sha256'];samples.append(value)
            native,big,sample=samples
            assert sample['raw_rows']==independent_rows(native['raw_rows'],big['raw_rows'],vp['word_augmentation'])
            mp=candidate/'scaled/maps'/(key(sid)+'.npz');npth=source/'export_train/maps'/(key(sid)+'.npz')
            assert file_hash(mp)==sample['maps_sha256'] and file_hash(npth)==native['maps_sha256']
            evidence[str(mp)]=sample['maps_sha256'];evidence[str(npth)]=native['maps_sha256']
            with np.load(mp,allow_pickle=False) as a,np.load(npth,allow_pickle=False) as b:
                assert np.array_equal(a['probabilities'],b['probabilities']) and np.array_equal(a['encoder_context'],context[sid])
            p=out/'samples'/(key(sid)+'.json')
            write(p,dict(sample,maps_path=str(mp),frontend_policy_sha256=policy_sha,
                source_checkpoint_sha256=selected['checkpoint_sha256'],source_observation_path=ci[sid]['path'],
                source_observation_sha256=ci[sid]['sha256'],decoder_fit_excluded=False,upstream_encoder_exposure=True))
            index[sid]=dict(path=str(p),sha256=file_hash(p))
        write(out/'index.json',index);write(out/'source_evidence.json',evidence)
        write(out/'independent_audit.json',dict(status='PASS',images=len(ids),index_sha256=file_hash(out/'index.json'),
            source_training_plan_sha256=file_hash(source/'resolved_configuration.json'),source_checkpoint_sha256=selected['checkpoint_sha256'],
            frontend_policy_sha256=policy_sha,source_candidate_audit_sha256=file_hash(candidate/'independent_map_audit.json'),
            source_evidence_sha256=file_hash(out/'source_evidence.json'),source_sha256=file_hash(__file__),ground_truth_used=False,
            scope='All original validation images, independent word admission and exact native map/context/RGB identities; full-FIT weights, no new neural calls. Not a fresh encoder holdout.'))
        snapshot=out/'qualification_code';snapshot.mkdir()
        for name in ('scripts/qualify_uniform_multiscale_validation_banks.py','scripts/build_uniform_multiscale_word_banks.py'):
            src=Path(__file__).resolve().parents[1]/name;dest=snapshot/src.name;shutil.copyfile(src,dest)
        print('MATCHING VALIDATION BANK PASS',ds,len(ids),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);run(p.parse_args().config)
