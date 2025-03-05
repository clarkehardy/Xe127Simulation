import os
import time
import numpy as np

# choose simulation options
first_seed = 21
num_sets = 20
num_evts = 5e4
run_name = "a50k"
alpha = True
if alpha==True:
    script = "run_job_alpha.sh"
    max_time = str(int(np.around(0.015*6.5*num_evts)))
else:
    script = "run_job.sh"
    max_time = str(int(np.around(0.015*num_evts)))

# determine seeds and run time
seeds = np.arange(first_seed,num_sets+first_seed)

# loop through and submit jobs
for seed in seeds:
    name = run_name+'_'+str(int(seed))
    os.system("sbatch -J "+name+" -t "+max_time+" -o slurm_out/slurm-%j.out "+script+" "+str(int(num_evts))+" "+str(int(seed))+" "+name)
    time.sleep(0.5)
