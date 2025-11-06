#!/bin/bash
#SBATCH --job-name=kid
#SBATCH --time=0-20:00:00

#SBATCH -p frida
#SBATCH -w aga1
#SBATCH -c 16
#SBATCH --gpus=A100:1
#SBATCH --output=clipscore.txt

srun \
  --container-mounts ${PWD}:${PWD} \
  --container-workdir ${PWD} \
  bash -c 'python3 clipscore.py'