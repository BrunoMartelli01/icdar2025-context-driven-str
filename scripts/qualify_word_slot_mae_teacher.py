"""Correct and qualify MAE pretraining inputs before any MAE-supervised fitting."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import shutil
import time
import traceback

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from recognition.word_slot_teachers import load_mae,character_vectors
from segmentation.word_slot_data import read,write,sha,sample_key,load_rgb
from scripts.run_word_slot_methods import setup,memory

REPO=Path(__file__).resolve().parents[1]


def qualify(root,config,config_path):
    protocol=read(root/'protocol.json');c=protocol['config'];config_sha=sha(config_path)
    completed=root/'mae_preprocessing_v2_complete.json'
    if completed.exists():
        if read(completed)['config_sha256']!=config_sha:raise ValueError('Changed MAE preprocessing contract')
        return
    if (root/'branches/mae_character').exists():
        raise ValueError('MAE branch already initialized; preserve it and fork before changing its teacher targets')
    if config['crop_size_hw']!=[32,128] or config['pil_interpolation']!='bicubic':raise ValueError('Unqualified pretraining input shape/interpolation')
    hardware=setup(c);device=c['runtime']['device'];teacher,loading=load_mae(c['mae'],device)
    destination=root/'teacher_cache'/'mae_pretraining_v2_pending';destination.mkdir(parents=True,exist_ok=True)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter();count=0;images=0;hashes={}
    with torch.no_grad():
        for phase in ('fit','calibration'):
            for number,record in enumerate(protocol['index'][phase],1):
                path=destination/(sample_key(record['sample_id'])+'.pt')
                if path.exists():
                    saved=torch.load(path,map_location='cpu',weights_only=True)
                    if saved['preprocessing_config_sha256']!=config_sha or saved['image_sha256']!=record['image_sha256']:
                        raise ValueError('Partial corrected MAE cache contract mismatch')
                else:
                    rgb=load_rgb(record);crops=[]
                    for word in record['ground_truth']:
                        crop=rgb.crop(tuple(map(int,word['bbox']))).resize((128,32),Image.Resampling.BICUBIC)
                        crops.append(TF.normalize(TF.to_tensor(crop),config['normalization_mean'],config['normalization_std']))
                    tokens=[]
                    for offset in range(0,len(crops),config['batch_size']):
                        output=teacher(torch.stack(crops[offset:offset+config['batch_size']]).to(device))[:,1:]
                        tokens.extend(output.cpu().split(1))
                    characters=[character_vectors(token,[min(len(word['text']),25)]) for token,word in zip(tokens,record['ground_truth'])]
                    if any(not torch.isfinite(value).all() for value in characters):raise FloatingPointError('Nonfinite MAE character targets')
                    torch.save({'characters':characters,'teacher_sha256':c['mae']['checkpoint_sha256'],
                          'image_sha256':record['image_sha256'],'preprocessing_config_sha256':config_sha,
                          'input_preprocessing':config,'alignment':'uniform horizontal bins, approximate'},path)
                    count+=len(crops)
                hashes[path.name]=sha(path);images+=1
                if number%100==0 or number==len(protocol['index'][phase]):print('MAE_PRETRAINING_TARGETS',phase,number,flush=True)
    expected=sum(len(protocol['index'][phase]) for phase in ('fit','calibration'))
    if len(hashes)!=expected:raise ValueError('Missing corrected MAE image targets')
    previous=root/'teacher_cache'/'mae_initial_imagenet_v1'
    if previous.exists():raise FileExistsError('Initial MAE target history already archived; inspect interrupted publication')
    old_audit=read(root/'teacher_cache_audit.json');write(root/'mae_preprocessing_v2_previous_teacher_audit.json',old_audit)
    (root/'teacher_cache'/'mae').rename(previous);destination.rename(root/'teacher_cache'/'mae')
    audit={'status':'PASS','config':config,'config_sha256':config_sha,'source_code_sha256':sha(Path(__file__)),
           'teacher_loading':loading,'runtime':hardware,'precision':'float32','models':['official MAE-pretrained ViT-S'],
           'batch_size':config['batch_size'],'input_shape':[config['batch_size'],3,*config['crop_size_hw']],
           'images':images,'new_crops':count,'sha256_by_sample_file':hashes,'seconds':time.perf_counter()-started,
           'previous_cache_preserved':str(previous),'mae_branch_not_started_before_publication':True,**memory()}
    old_audit['mae']=audit;write(root/'teacher_cache_audit.json',old_audit);write(completed,audit)
    del teacher;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',required=True,choices=['approximate'])
    parser.add_argument('--config',required=True,type=Path);parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args();config=read(args.config);root=args.run.resolve()
    if REPO/'configs/approximate' not in args.config.resolve().parents or config['reproduction_mode']!=args.mode or config['experiment_kind']!='word_slot_mae_pretraining_preprocessing_v2':raise ValueError('Explicit APPROXIMATE teacher qualification config required')
    try:qualify(root,config,args.config)
    except Exception as error:
        write(root/('mae_preprocessing_failure_'+str(time.time_ns())+'.json'),{'status':'FAILED','type':type(error).__name__,
              'message':str(error),'traceback':traceback.format_exc(),'oom':isinstance(error,torch.cuda.OutOfMemoryError)});raise
