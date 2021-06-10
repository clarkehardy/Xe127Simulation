#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=120
#SBATCH --job-name=lm_recon
#SBATCH --export=ALL
#SBATCH --output=../outputs/lm-%A_%a.out
#SBATCH --error=../outputs/lm-%A_%a.err
#SBATCH --array=152,160

./lm_recon${SLURM_ARRAY_TASK_ID}.sh
