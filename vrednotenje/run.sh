#!/bin/bash
#SBATCH --job-name=kid
#SBATCH --time=0-20:00:00

#SBATCH -p frida
#SBATCH -w ana
#SBATCH -c 16
#SBATCH --gpus=A100_80GB:1
#SBATCH --output=ucnamnozica.txt

srun \
  --container-mounts ${PWD}:${PWD} \
  --container-workdir ${PWD} \
  bash -c 'python3 fidistance.py'