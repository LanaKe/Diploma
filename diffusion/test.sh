#!/bin/bash
#SBATCH --job-name=diff
#SBATCH --time=0-47:00:00

#SBATCH -p frida
#SBATCH -w ana
#SBATCH -c 64
#SBATCH --gpus=A100:0
#SBATCH --output=test3.txt

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

srun \
  --container-mounts ${PWD}:${PWD} \
  --container-workdir ${PWD} \
  bash -c 'python3 train.py'