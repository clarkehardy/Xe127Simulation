#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=5
#SBATCH --job-name=sim_proc
#SBATCH --export=ALL
#SBATCH --output=proc-%A_%a.out
#SBATCH --error=proc-%A_%a.err
#SBATCH --array=101-500

SEED_NUM=`printf %d $SLURM_ARRAY_TASK_ID`

python3 GammaDataProcessing.py -output_dir $SIM_DIR -num_events 10000 -input_file $SIM_DIR/ExternalGammas_light-Co60-5000-seed${SEED_NUM}.root
