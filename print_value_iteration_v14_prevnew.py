import os
import sys
import numpy as np 

algo = sys.argv[1]
iter = sys.argv[2]

input_dir = os.path.join('logs', algo)

test_list = ['renderonly_test_stage1_oldtask_'+str(iter)]
test_list += ['renderonly_test_stage1_newtask_'+str(iter)]


for render_dir in test_list:
    psnr_path = os.path.join(input_dir, render_dir, 'psnr.txt')
    if not os.path.exists(psnr_path):
        print(render_dir)
        continue 

    with open(psnr_path, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split('[')[1].split(']')[0]
            print(line, end='\t')
            
print()
