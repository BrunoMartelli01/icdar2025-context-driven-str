"""Ground-truth-free live U-Net/PARSeq specialist with the trained seed protocol."""
from __future__ import annotations

import gc
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from models.word_slot_unet import sampling_grid,deterministic_sample
from recognition.word_slot_evaluation import nms
from scripts.run_word_slot_methods import make_model,load_prior,load_delta,reader
from segmentation.word_slot_data import read,sha,write,image_tensor,canvas_to_original
from segmentation.totaltext_word_kernel import expand_word_kernels,instance_proposals
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask


class WordSlotSpecialist:
    """Use this adapter for trained inference; model.forward is unseeded research.

    The same GT-blind predicted initialization used in training/evaluation is
    computed live. No annotation, owner map, cache or dataset manifest is read.
    Call the declared runtime setup once before construction on CUDA.
    """
    def __init__(self,run,arm,*,mode):
        if mode!='approximate':raise ValueError('Word-slot specialization is an isolated APPROXIMATE research path')
        self.run=Path(run);protocol=read(self.run/'protocol.json');self.config=protocol['config'];self.arm=arm
        if self.config['reproduction_mode']!=mode or arm not in self.config['arms']:raise ValueError('Unqualified branch/mode')
        complete=read(self.run/'branches'/arm/'complete.json');checkpoint=self.run/'branches'/arm/'last.pt'
        if complete['status']!='SUCCESS' or complete['epochs']!=self.config['training']['branch_epochs'] or sha(checkpoint)!=complete['checkpoint_sha256']:
            raise ValueError('A completed, hash-identified branch with its declared budget is required')
        if not torch.are_deterministic_algorithms_enabled():raise RuntimeError('Configure the declared deterministic runtime before live inference')
        self.checkpoint_sha256=sha(checkpoint);self.device=self.config['runtime']['device'];tick=time.perf_counter()
        self.model=make_model(self.config,arm);self.initialization=load_prior(self.model,self.config);state=load_delta(self.model,checkpoint,self.config)
        if state['arm']!=arm:raise ValueError('Decoder checkpoint branch metadata mismatch')
        self.model.eval().to(self.device);self.recognizer=reader(self.config);torch.cuda.synchronize()
        self.loading_seconds=time.perf_counter()-tick

    def predict(self,image_path,*,artifact_directory=None):
        if self.model is None or self.recognizer is None:raise RuntimeError('Released specialist; construct it again before predicting')
        path=Path(image_path).resolve();image_sha=sha(path)
        with Image.open(path) as opened:
            wh=opened.size;orientation=opened.getexif().get(274,1) or 1;rgb=opened.convert('RGB')
        c=self.config;device=self.device;tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            pyramid,pixel=self.model.prior_pyramid(image_tensor(rgb).to(device))
            probability=pixel[0].sigmoid().cpu().numpy();initial=c['initialization']
            canvas=np.asarray(letterbox_canvas_mask(wh,native_size=probability.shape[-1]),bool)
            owners,_=expand_word_kernels((probability[0]>=initial['text_threshold'])&canvas,
                                          (probability[1]>=initial['kernel_threshold'])&canvas)
            proposals=instance_proposals(owners)
            seeds=torch.tensor([p['bbox_native'] for p in proposals[:initial['maximum_predicted_seeds']]],
                               dtype=torch.float32,device=device).reshape(-1,4)/probability.shape[-1]
            output=self.model.decode_pyramid(pyramid,prior_boxes=seeds)
            score=output['presence'].sigmoid();boxes=output['boxes']
            ids=nms(boxes.cpu().tolist(),score.cpu().tolist(),c['evaluation']['nms_iou'],c['evaluation']['ocr_budget'])
            boxes=boxes[ids];original=canvas_to_original(boxes,wh);normalized=original/original.new_tensor(list(wh)*2)
            offsets=output['strip_offsets'][ids] if c['arms'][self.arm].get('strip') else None
            grid=sampling_grid(normalized,offsets,height=32,width=128,strength=c['training']['strip_strength'])
            crops=deterministic_sample(TF.to_tensor(rgb),grid,device)*2-1
            if len(ids):
                text,confidence=self.recognizer.model.tokenizer.decode(self.recognizer.model(crops).softmax(-1))
            else:text=[];confidence=[]
            predictions=[]
            for index,(word,prob) in enumerate(zip(text,confidence)):
                predictions.append({'text':word,'bbox':original[index].cpu().tolist(),'presence':float(score[ids[index]]),
                       'confidence':float(prob.prod()),'mean_token_confidence':float(prob.mean()) if len(prob) else 0.,
                       'slot_index':ids[index],'crop_tensor_sha256':hashlib.sha256(crops[index].cpu().numpy().tobytes()).hexdigest()})
            torch.cuda.synchronize();seconds=time.perf_counter()-tick
            intermediate=None
            if artifact_directory is not None:
                artifact_directory=Path(artifact_directory);artifact_directory.mkdir(parents=True,exist_ok=True)
                intermediate=artifact_directory/'student_intermediates.npz'
                if intermediate.exists():raise FileExistsError('Preserve existing live intermediates')
                masks=F.interpolate(output['masks'][None,ids],(c['training']['mask_size'],)*2,mode='nearest')[0]
                np.savez_compressed(intermediate,selected_mask_logits=masks.cpu().numpy(),boxes_canvas=boxes.cpu().numpy(),
                                    rgb_grid=grid.cpu().numpy(),prior_pixel_logits=pixel.cpu().numpy())
        return {'schema_version':1,'reproduction_mode':'approximate','claim_label':'approximate reproduction',
                'result_class':'single-image live smoke / research inference','arm':self.arm,
                'input':{'path':str(path),'sha256':image_sha,'image_wh':list(wh),'orientation':orientation},
                'predictions':predictions,'ground_truth_used':False,'cached_prior_used':False,
                'decoder_checkpoint_sha256':self.checkpoint_sha256,'pixel_prior_sha256':c['input_sha256']['source_checkpoint'],
                'parseq_provenance':dict(self.recognizer._provenance),'intermediate_path':str(intermediate) if intermediate else None,
                'timing':{'model_loading_seconds':self.loading_seconds,'live_prior_decoder_ocr_seconds':seconds},
                'memory':{'precision':'float32','device':device,'image_input_shape':[1,3,512,512],
                          'ocr_input_shape':[len(ids),3,32,128],'models':['frozen qualified pixel U-Net','trained word-slot decoder','frozen full RGB PARSeq'],
                          'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'oom':False}}

    def release(self):
        # Lightning may keep references to its PARSeq submodules after the
        # adapter is released. Move weights off CUDA explicitly before dropping
        # our references; cleanup happens only after inference has finished.
        if self.recognizer is not None and self.recognizer.model is not None:self.recognizer.model.cpu()
        if self.model is not None:self.model.cpu()
        if self.recognizer is not None:self.recognizer.release()
        self.recognizer=None;self.model=None;gc.collect();torch.cuda.empty_cache()
        return {'student_released':True,'student_weights_moved_to_cpu_before_release':True,
                'remaining_allocated_bytes':torch.cuda.memory_allocated()}
