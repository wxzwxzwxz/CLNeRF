import os 
import sys  
import cv2
import numpy as np

testset_list = ['renderonly_test_stage1_bookshelf_209999', \
    'renderonly_test_stage1_path_209999', \
    'renderonly_test_stage1_sofa1_209999', \
    'renderonly_test_stage1_sofa2_209999', \
    'renderonly_test_stage1_sofa3_209999', \
    'renderonly_test_stage1_table_209999']

algo_ours = 'blendswap_whitehouse_origin_v11apple_table_finetune_on_origin_v11_near1_far10_20w_tea1024_wtea0.1_lrgs_ts8'
algo_comparison = 'blendswap_whitehouse_origin_v11apple_table_finetune_on_origin_v11_near1_far10_20w_tea1024_wtea0.1_adav0_adadim16_adalr5e-4_ts8_v3'
output_dir = 'vis_comparison'
output_path = os.path.join(output_dir, 'ts8_ours_vs_adaptor')
try:
    os.makedirs(output_path)
except:
    pass

for testset in testset_list:
    print(testset)
    try:
        cur_output_path = os.path.join(output_path, testset)
        os.makedirs(cur_output_path)
    except:
        pass

    input_ours = os.path.join('logs', algo_ours, testset)
    input_comparison = os.path.join('logs', algo_comparison, testset)
    input_file_list = os.listdir(input_ours)
    for input_file in input_file_list:
        if '.jpg' in input_file:
            img_ours = cv2.imread(os.path.join(input_ours, input_file))
            img_comparison = cv2.imread(os.path.join(input_comparison, input_file))
            img_output = np.concatenate([img_ours, img_comparison], 0)

            cv2.imwrite(os.path.join(cur_output_path, input_file), img_output)
            