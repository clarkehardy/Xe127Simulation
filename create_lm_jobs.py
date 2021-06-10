import subprocess

args = []
with open('jobs_list.txt','r') as infile:
    for line in infile:
        args.append(line)

for i in range(len(args)):
    with open('lm_recon'+str(i+1)+'.sh','w') as newfile:
        newfile.write('python3 roundtrip.py '+args[i][:-1]+' -train -input_files $SIM_DIR/g4andcharge_10kevts_xe127_seed*_REDUCED.pkl')

subprocess.Popen('chmod +x lm_recon*.sh',shell=True).wait()
