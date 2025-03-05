import subprocess

first_num = 0

args = []
with open('corr_jobs_list.txt','r') as infile:
    for line in infile:
        args.append(line)

for i in range(len(args)):
    with open('lm_corr'+str(i+first_num)+'.sh','w') as newfile:
        newfile.write('python3 add_lms.py '+args[i][:-1])

subprocess.Popen('chmod +x lm_corr*.sh',shell=True).wait()
