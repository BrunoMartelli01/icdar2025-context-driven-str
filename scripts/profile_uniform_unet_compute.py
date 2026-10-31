"""Count profiler-supported FLOPs of the qualified U-Net on CPU, without GT."""

import argparse
import copy
from pathlib import Path
import platform
import time

import torch
from torch.profiler import profile, ProfilerActivity

from scripts.export_uniform_unet import checked_training, model
from scripts.run_uniform_unet_training import write
from segmentation.uniform_unet_fusion import file_hash


def run(args):
    plan, complete = checked_training(args.run_root)
    plan = copy.deepcopy(plan); plan['config']['device'] = 'cpu'
    torch.set_num_threads(args.cpu_threads)
    net = model(plan, args.run_root).eval(); results = []
    for size in args.input_size:
        if size < 32 or size % 32:
            raise ValueError('Input sizes must be positive multiples of32')
        x = torch.zeros((1, 3, size, size), dtype=torch.float32)
        started = time.perf_counter()
        with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU], with_flops=True) as trace:
            y = net(x)
        if tuple(y.shape) != (1, 2, size//2, size//2) or not torch.isfinite(y).all():
            raise ValueError('Wrong real U-Net output in compute qualification')
        operators = [dict(operator=row.key, calls=row.count, flops=int(row.flops))
            for row in trace.key_averages() if row.flops]
        results.append(dict(input_bchw=list(x.shape), output_bchw=list(y.shape),
            counted_flops=sum(row['flops'] for row in operators), operators=operators,
            counted_convolution_flops=sum(row['flops'] for row in operators if row['operator'] == 'aten::conv2d'),
            cpu_profile_wall_seconds=time.perf_counter()-started))
    write(args.output, dict(status='COMPLETE', mode='approximate', result_class='smoke-test result',
        claim_label='approximate reproduction', checkpoint_sha256=complete['checkpoint_sha256'],
        parameters=sum(p.numel() for p in net.parameters()), device='cpu', precision='float32',
        cpu_threads=args.cpu_threads, platform=platform.platform(), torch=torch.__version__,
        source_sha256=file_hash(__file__), results=results,
        counting_protocol='All nonzero FLOP events reported by PyTorch profiler with_flops=True: conv2d and elementwise add/mul in this network. Convolution multiply and add count separately. Unreported operators and memory traffic excluded; zero RGB tensor, batch1, inference only. See exact per-operator counts.',
        scientific_choices=dict(HARDWARE_DEVIATION='CPU FP32 engineering shape/compute qualification of the same strict-loaded GPU deployment checkpoint; no change to production placement.',
            ASSUMPTION='Declared partial-operation FLOP convention and synthetic zero input; no PARSeq/DeepSolo or backward pass included.',
            UNRESOLVED='Paper FLOP convention equivalence, complete pipeline FLOPs and real idle-GPU latency remain unresolved.'),
        ground_truth_used=False, gpu_latency_measured=False))
    print('UNIFORM UNET COMPUTE', [(r['input_bchw'], r['counted_flops']) for r in results], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--input-size', type=int, action='append', required=True)
    p.add_argument('--cpu-threads', type=int, required=True); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
