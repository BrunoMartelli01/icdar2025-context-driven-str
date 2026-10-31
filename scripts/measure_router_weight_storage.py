"""Read-only storage/tensor-size comparison; no GPU execution or precision change."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct

import joblib
import torch


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=json.loads(a.config.read_text());paths=c['paths'];groups=defaultdict(lambda:dict(tensor_bytes=0,tensor_elements=0,tensors=0));blip_files=[]
    for file in sorted(Path(paths['blip_snapshot']).glob('model-*.safetensors')):
        with file.open('rb') as stream:
            n=struct.unpack('<Q',stream.read(8))[0];header_bytes=stream.read(n);header=json.loads(header_bytes)
        blip_files.append(dict(path=str(file),file_bytes=file.stat().st_size,header_sha256=hashlib.sha256(header_bytes).hexdigest()))
        for name,value in header.items():
            if name=='__metadata__':continue
            prefix=name.split('.')[0];elements=1
            for s in value['shape']:elements*=s
            g=groups[prefix];g['tensor_bytes']+=value['data_offsets'][1]-value['data_offsets'][0];g['tensor_elements']+=elements;g['tensors']+=1
    clip_path=Path(paths['clip_checkpoint']);clip=torch.load(clip_path,map_location='meta',mmap=True,weights_only=True)
    clip_bytes=sum(v.numel()*v.element_size() for v in clip.values() if isinstance(v,torch.Tensor))
    head_path=Path(paths['heavy_head']);heavy=torch.load(head_path,map_location='meta',mmap=True,weights_only=True)
    heavy_bytes=sum(v.numel()*v.element_size() for v in heavy['state_dict'].values())
    light_path=Path(paths['pruned_run'])/'selected_deployment.joblib';light=joblib.load(light_path)['bundle']
    coefficient_bytes=light['regressor'].coef_.nbytes+light['regressor'].intercept_.nbytes
    normalize_bytes=light['scaler'].mean_.nbytes+light['scaler'].scale_.nbytes
    active_prefixes=['vision_model','qformer','query_tokens','language_projection']
    active_bytes=sum(groups[k]['tensor_bytes'] for k in active_prefixes)
    result=dict(status='PASS',mode='approximate',units='Decimal bytes/kB/MB/GB; tensors preserve saved dtype. Weight-only sizes exclude activations and framework/process overhead.',
        essential=dict(head_coefficients=int(light['regressor'].coef_.size+1),coefficient_bytes=coefficient_bytes,
            mean_and_scale_bytes=normalize_bytes,numerical_inference_bytes=coefficient_bytes+normalize_bytes,
            unused_stored_scaler_variance_bytes=light['scaler'].var_.nbytes,deployment_file_bytes=light_path.stat().st_size,
            deployment_sha256=sha(light_path),additional_foundation_weight_bytes=0),
        heavy=dict(head_parameters=sum(v.numel() for v in heavy['state_dict'].values()),head_tensor_bytes=heavy_bytes,
            head_file_bytes=head_path.stat().st_size,clip_tensor_bytes=clip_bytes,clip_file_bytes=clip_path.stat().st_size,
            blip_full_tensor_bytes=sum(v['tensor_bytes'] for v in groups.values()),blip_full_file_bytes=sum(v['file_bytes'] for v in blip_files),
            blip_groups=dict(groups),blip_files=blip_files,
            current_full_loaded_weight_bytes=heavy_bytes+clip_bytes+sum(v['tensor_bytes'] for v in groups.values()),
            theoretical_visual_only_weight_bytes=heavy_bytes+clip_bytes+active_bytes,
            visual_only_scope='Theoretical subset for the existing visual() operation; actual qualified loader constructs complete BLIP2 including T5. No pruning, loading change, new inference or equivalent-output claim is performed here.'),
        shared_expert_scope='U-Net/PARSeq/GNN/RGB-verifier/text-editor and DeepSolo fallback are unchanged and excluded from incremental router weights. All saved expert-generation costs remain.',
        current_runtime_scope='CLIP/BLIP generation used sequential stages. Their summed checkpoint/weight size is not a measured simultaneous VRAM peak.',
        no_new_gpu_execution=True,analysis_code_sha256=sha(Path(__file__)),config_sha256=sha(a.config))
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='PASS',essential_numeric_bytes=coefficient_bytes+normalize_bytes,essential_file_bytes=light_path.stat().st_size,
                         heavy_head_bytes=heavy_bytes,heavy_total_GB=result['heavy']['current_full_loaded_weight_bytes']/1e9,
                         heavy_visual_only_GB=result['heavy']['theoretical_visual_only_weight_bytes']/1e9,blip_groups=dict(groups))))


if __name__=='__main__':main()
