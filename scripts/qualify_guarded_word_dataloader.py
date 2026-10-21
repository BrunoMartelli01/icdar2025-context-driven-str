"""Real CPU DataLoader boundary smoke for the opt-in V3 training dataset."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from segmentation.data import file_sha256
from segmentation.guarded_word_dataset import make_audited_guarded_dataset


def run(args):
    if args.output.exists():raise FileExistsError('Use a fresh smoke receipt')
    torch.set_num_threads(4)
    dataset=make_audited_guarded_dataset(args.config,augmentation_enabled=True,verify_checksums=False)
    dataset.transform.set_epoch(args.epoch)
    by_id={s.sample_id:i for i,s in enumerate(dataset.manifest.samples)}
    indices=[by_id[sid] for sid in args.sample_ids]
    loader=DataLoader(Subset(dataset,indices),batch_size=args.batch_size,num_workers=args.num_workers,
        shuffle=False,persistent_workers=False)
    rows=[]
    for batch in loader:
        for offset,sid in enumerate(batch['sample_id']):
            direct=dataset[by_id[sid]]
            for key in ['image','mask','validity','separation','native_owners']:
                if not torch.equal(batch[key][offset],direct[key]):
                    raise AssertionError('Worker/direct mismatch: '+key)
            audit=json.loads(batch['gt_guard_audit_json'][offset])
            if audit['sample_id']!=sid or audit['epoch']!=args.epoch or audit!=json.loads(direct['gt_guard_audit_json']):
                raise AssertionError('Per-sample guard evidence lost at worker boundary')
            if batch['target_recipe_sha256'][offset]!=file_sha256(args.config):
                raise AssertionError('Wrong target recipe provenance')
            rows.append({'sample_id':sid,'native_owner_array_sha256':sha256(batch['native_owners'][offset].numpy().astype('<i4').tobytes()).hexdigest(),
                'gt_guard_stats':audit['stats']})
    if [r['sample_id'] for r in rows]!=args.sample_ids:
        raise AssertionError('DataLoader dropped/reordered samples')
    report={'status':'PASS','claim_label':'approximate reproduction','result_class':'smoke-test result',
        'device':'cpu','new_training':False,'epoch':args.epoch,'num_workers':args.num_workers,
        'batch_size':args.batch_size,'input_shape':[args.batch_size,3,256,256],'target_shape':[args.batch_size,1,128,128],
        'code_sha256':{str(p):file_sha256(p) for p in [Path(__file__),Path('segmentation/guarded_word_dataset.py')]},
        'config_sha256':file_sha256(args.config),'samples':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'PASS','samples':len(rows),'batch_size':args.batch_size,'num_workers':args.num_workers}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epoch',type=int,required=True)
    p.add_argument('--num-workers',type=int,required=True);p.add_argument('--batch-size',type=int,required=True)
    p.add_argument('--sample-ids',nargs='+',required=True);run(p.parse_args())
