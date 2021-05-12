#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=75
#SBATCH --job-name=xe127_sim
#SBATCH --export=ALL
#SBATCH --output=xe127-%A_%a.out
#SBATCH --error=xe127-%A_%a.err
#SBATCH --array=1-1000

SEED_NUM=`printf %03d $SLURM_ARRAY_TASK_ID`

./Xe127Sim.sh $SEED_NUM
