#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=5
#SBATCH --job-name=sim_proc
#SBATCH --export=ALL
#SBATCH --output=proc-%A_%a.out
#SBATCH --error=proc-%A_%a.err
#SBATCH --array=1-1000

SEED_NUM=`printf %03d $SLURM_ARRAY_TASK_ID`

python3 DataProcessing.py -output_dir $SIM_DIR -num_events 10000 -input_file $SIM_DIR/g4andcharge_10kevts_xe127_seed${SEED_NUM}_10msEL.root
