"""Standalone native encoder worker; no evaluator/dependency-stack imports."""
import argparse

from recognition.clip_specialized_runtime import worker


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity', required=True)
    parser.add_argument('--device', required=True, choices=['cuda:0','cpu'])
    args = parser.parse_args()
    worker(args.identity,args.device)
