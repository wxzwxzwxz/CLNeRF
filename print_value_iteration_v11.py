import os
import sys
import numpy as np 

# blendswap_whitehouse_origin_v11apple_table_finetune_on_origin_v11_near1_far10_40w_ts8
algo = sys.argv[1]
iter = sys.argv[2]

input_dir = os.path.join('logs', algo)
print(algo)

test_list = ['renderonly_test_path_'+str(iter)]
test_list += ['renderonly_test_bookshelf_'+str(iter)]
test_list += ['renderonly_test_sofa1_'+str(iter), 'renderonly_test_sofa2_'+str(iter), 'renderonly_test_sofa3_'+str(iter)]
test_list += ['renderonly_test_table_'+str(iter)]

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
print()
