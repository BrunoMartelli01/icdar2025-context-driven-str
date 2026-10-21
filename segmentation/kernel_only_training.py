"""Only Kernel and validity enter this supervised loss."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import math

import torch
from segmentation.training import SegmentationTrainingConfig
from segmentation.word_multitask import masked_bce_dice


@dataclass(frozen=True)
class KernelTrainingConfig(SegmentationTrainingConfig):
    loss: str = 'masked_BCE_Dice_kernel_only_v1'

    def __post_init__(self):
        if self.loss != 'masked_BCE_Dice_kernel_only_v1' or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('Invalid Kernel-only training configuration')
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError('Invalid Kernel batch size')
        values=asdict(self);values.update(loss='masked_BCEWithLogitsLoss',learning_rate=1e-4,batch_size=8)
        SegmentationTrainingConfig(**values)


def kernel_loss(logits, batch, options):
    return masked_bce_dice(logits,batch['kernel'],batch['validity'],**options)


def train_kernel_epoch(model,batches,optimizer,device,config):
    model.train();total=count=steps=skipped=0;ids=[]
    for index,batch in enumerate(batches,1):
        image=batch['image'].to(device)
        target={k:batch[k].to(device) for k in ('kernel','validity')}
        if image.device != device:raise RuntimeError('Kernel training input left requested device')
        optimizer.zero_grad(set_to_none=True);logits=model(image)
        if logits.device != image.device:raise RuntimeError('Kernel output device mismatch')
        loss=kernel_loss(logits,target,config['loss'])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite Kernel-only loss')
        if target['validity'].any():
            loss.backward();optimizer.step();steps+=1
        else:skipped+=1
        total+=float(loss.detach())*len(image);count+=len(image);ids.extend(batch['sample_id'])
        if index==1 or index%25==0:
            print('[Kernel-only] batch={} steps={} device={} loss={:.5f}'.format(index,steps,logits.device,float(loss.detach())),flush=True)
    if not count:raise ValueError('Empty Kernel training population')
    return {'loss':total/count,'kernel_loss':total/count,'active_loss_components':['kernel'],
            'samples':count,'steps':steps,'skipped_all_invalid_batches':skipped,'compute_device':str(device),
            'sample_order_sha256':sha256(''.join(s+'\n' for s in ids).encode()).hexdigest()}
