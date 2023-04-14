import os
import sys
import numpy as np 

algo = sys.argv[1]

input_dir = os.path.join('logs', algo)
# model_list = os.listdir(input_dir)
# model_list = [for model in model_list if '.tar' in model]

# render_list = os.listdir(input_dir)
# render_list = sorted(render_list)
print(algo)

test_list = ['renderonly_test_199999']
test_list += ['renderonly_test_path_199999_path']
test_list += ['renderonly_test_sofa1_199999_sofa1', 'renderonly_test_sofa2_199999_sofa2', 'renderonly_test_sofa3_199999_sofa3']
test_list += ['renderonly_test_table_199999_table']

test_list += ['renderonly_test_399999']
test_list += ['renderonly_test_path_399999_path']
test_list += ['renderonly_test_sofa1_399999_sofa1', 'renderonly_test_sofa2_399999_sofa2', 'renderonly_test_sofa3_399999_sofa3']
test_list += ['renderonly_test_table_399999_table']

for render_dir in test_list:
    psnr_path = os.path.join(input_dir, render_dir, 'psnr.txt')
    if not os.path.exists(psnr_path):
        print(render_dir)
        continue 
    # print(psnr_path)
    # print(render_dir.split('renderonly_test_')[-1], end='\t')
    with open(psnr_path, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split('[')[1].split(']')[0]
            print(line, end=', ')
print()
print()
