import os
import sys
import numpy as np 

algo = sys.argv[1]

input_dir = os.path.join('logs', algo)
# model_list = os.listdir(input_dir)
# model_list = [for model in model_list if '.tar' in model]

render_list = os.listdir(input_dir)
render_list = sorted(render_list)
print(algo)

for render_dir in render_list:
    if 'renderonly_test' not in render_dir or 'v7v8' not in render_dir:
        continue

    psnr_path = os.path.join(input_dir, render_dir, 'psnr.txt')
    if not os.path.exists(psnr_path):
        continue 
    # print(render_dir.split('renderonly_test_')[-1], end='\t')
    with open(psnr_path, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split('[')[1].split(']')[0]
            print(line, end=', ')
print()
print()

for render_dir in render_list:
    if 'renderonly_test' not in render_dir or 'v7v8' in render_dir:
        continue

    psnr_path = os.path.join(input_dir, render_dir, 'psnr.txt')
    if not os.path.exists(psnr_path):
        continue 
    # print(render_dir.split('renderonly_test_')[-1], end='\t')
    with open(psnr_path, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split('[')[1].split(']')[0]
            print(line, end=', ')
print()
print()