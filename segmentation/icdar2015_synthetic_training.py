"""Explicit AdamW stages and counted synthetic supervision; approximate only."""

from copy import deepcopy
from dataclasses import asdict,dataclass
from itertools import islice
import json

import torch

from segmentation.data import file_sha256
from segmentation.icdar2015_improvement_runtime import REPO
from segmentation.icdar2015_target70 import Target70TrainingConfig,training_config as adam_config,load_profile
from segmentation.training import load_checkpoint


def stage_config(config,phase):
    c=deepcopy(config);b=c['branches'][phase];c['phase']=phase
    c['training'].update(epochs=b['epochs'],learning_rate=b['decoder_learning_rate'])
    return c


@dataclass(frozen=True)
class SyntheticTrainingConfig(Target70TrainingConfig):
    optimizer:str='AdamW'

    def __post_init__(self):
        if self.optimizer!='AdamW':raise ValueError('Explicit AdamW research stage required')
        if self.checkpoint_selection not in ('last','validation_word_f1'):raise ValueError('Explicit last/target-word selection required')
        fields=asdict(self);fields.update(optimizer='Adam',checkpoint_selection='validation_word_f1');Target70TrainingConfig(**fields)


def training_config(c,b):
    if c['training']['optimizer']!='AdamW':raise ValueError('Profile must declare the actual AdamW optimizer')
    compatible=deepcopy(c);compatible['training']['optimizer']='Adam'
    fields=asdict(adam_config(compatible,b));fields['optimizer']='AdamW'
    fields['checkpoint_selection']='last' if c['phase']=='pretrain' else 'validation_word_f1'
    return SyntheticTrainingConfig(**fields)


def optimizer(model,c,b):
    t=c['training'];decoder=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.')]
    encoder=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')]
    return torch.optim.AdamW([
        dict(params=decoder,lr=t['learning_rate'],initial_lr=t['learning_rate'],weight_decay=t['weight_decay'],group_name='decoder'),
        dict(params=encoder,lr=b['encoder_learning_rate'],initial_lr=b['encoder_learning_rate'],weight_decay=b['encoder_weight_decay'],group_name='encoder_layer3_layer4')],
        betas=tuple(t['adam_betas']),eps=t['adam_epsilon'],foreach=t['optimizer_foreach'])


def load_source(model,args,c,synthetic_sha):
    report=json.loads(args.source_report.read_text());phase=args.branch
    if report['status']!='SUCCESS':raise ValueError('Successful source report required')
    for name,h in report['run_contract']['source_sha256'].items():
        if file_sha256(REPO/name)!=h:raise ValueError('Source scientific implementation changed: '+name)
    if phase=='pretrain':
        expected=[c[k] for k in ['source_checkpoint_sha256','source_run_manifest_sha256','source_validation_report_sha256']]
        actual=[file_sha256(p) for p in [args.source_checkpoint,args.source_report,args.source_validation]]
        if actual!=expected or not report['training_complete']:raise ValueError('Qualified c467 source bytes changed')
        p=REPO/c['source_training_profile']
        if file_sha256(p)!=c['source_training_profile_sha256']:raise ValueError('Qualified source profile changed')
        old=load_profile(p);old_branch=report['run_contract']['branch'];expected_training=adam_config(old,old['branches'][old_branch])
    else:
        rc=report['run_contract']
        if (rc['branch']!='pretrain' or rc['config_sha256']!=file_sha256(args.config)
                or rc['synthetic_manifest_sha256']!=synthetic_sha
                or file_sha256(args.source_checkpoint)!=report['checkpoint']['sha256']
                or file_sha256(args.source_validation)!=report['selected_validation_words_sha256']):
            raise ValueError('Matching pretraining checkpoint/provenance required')
        if not report['training_complete'] and not (args.smoke_steps and report['result_class']=='smoke-test result'):
            raise ValueError('Incomplete pretraining cannot enter full target fine-tuning')
        if report['training_complete'] and (report['completed_epochs']!=c['branches']['pretrain']['epochs'] or report['checkpoint']['epoch']!=report['completed_epochs']):
            raise ValueError('Complete last-epoch synthetic source required')
        pc=stage_config(c,'pretrain');expected_training=training_config(pc,pc['branches']['pretrain'])
    return load_checkpoint(args.source_checkpoint,model=model,expected_architecture=model.architecture_identity,
        expected_run_contract=report['run_contract'],expected_training_config=expected_training)


class CountedBatches:
    def __init__(self,loader,limit_microbatches=None):
        self.loader=loader;self.limit=limit_microbatches;self.counts={}

    def __iter__(self):
        stream=iter(self.loader) if self.limit is None else islice(self.loader,self.limit)
        for batch in stream:
            for key in ['visible_words','native_words_lost','outside_image_words']:
                if key in batch:self.counts[key]=self.counts.get(key,0)+int(batch[key].sum())
            yield batch
