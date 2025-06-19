#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --time=120
#SBATCH --job-name=lm_recon
#SBATCH --export=ALL
#SBATCH --output=../outputs/lm-%A_%a.out
#SBATCH --error=../outputs/lm-%A_%a.err
#SBATCH --array=679-703

./lm_recon${SLURM_ARRAY_TASK_ID}.sh
