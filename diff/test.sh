#!/bin/bash
#SBATCH --job-name=diff2
#SBATCH --time=0-47:00:00

#SBATCH -p frida
#SBATCH -w aga1
#SBATCH -c 64
#SBATCH --gpus=A100:1
#SBATCH --output=maska.txt

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

srun \
  --container-mounts ${PWD}:${PWD} \
  --container-workdir ${PWD} \
  bash -c 'python3 tutorial.py'