#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --time=0-18:00:00

#SBATCH -p frida
#SBATCH -w ixh
#SBATCH -c 64
#SBATCH --gpus=H100:1
#SBATCH --output=maska.txt

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

srun \
  --container-mounts ${PWD}:${PWD} \
  --container-workdir ${PWD} \
  bash -c 'python3 tutorial.py'