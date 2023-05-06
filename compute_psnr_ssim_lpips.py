import os
import sys
import cv2
import math
import lpips
import argparse
import numpy as np
from pathlib import Path
from ssim import ssim, ssim_exact
from tqdm import tqdm, trange

img2mse_np = lambda x, y : np.mean((x - y) ** 2)
mse2psnr_np = lambda x : -10. * np.log(x) / np.log(np.array([10.]))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='logs')
    parser.add_argument('--output_dir', type=str, default='output_eval')
    # parser.add_argument('--video_seq', type=str)
    parser.add_argument('--algo', type=str)
    parser.add_argument('--metric', type=str)
    args = parser.parse_args()
    return args 

def load_data(args):
    input_dir = os.path.join(args.input_dir, args.video_seq, args.algo)
    gt_input_dir = os.path.join(args.input_dir, args.video_seq, "gt")
    output_dir = os.path.join(args.output_dir, args.video_seq, args.algo)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    input_file_list = os.path.listdir(input_dir)
    return input_file_list

def compute_psnr(pred, target):
    # mse = np.mean((pred - target) ** 2)
    # if mse == 0: 
    #     return 100
    # max_pixel = 255.0
    # psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

    psnr = mse2psnr_np(img2mse_np(pred / 255.0, target / 255.0))
    return psnr

def compute_psnr_with_mask(pred, target, mask):
    mse = np.sum((pred * mask - target * mask) ** 2) / np.sum(mask)
    if mse == 0: 
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr
    
def compute_ssim(pred, target):
    return ssim_exact(np.array(pred).astype(np.float32)/255, np.array(target).astype(np.float32)/255)

def compute_lpips(pred_path, loss_fn):
    # Load images
    # img0 = lpips.im2tensor(lpips.load_image(pred_path)).cuda()
    # img1 = lpips.im2tensor(lpips.load_image(target_path)).cuda()
    
    # img0 = lpips.load_image(pred_path)
    # img1 = lpips.load_image(target_path)
    ori_image = lpips.load_image(pred_path)
    h, w, _ = ori_image.shape
    img0 = ori_image[:, :w//4, :]
    img1 = ori_image[:, w//4:w//2, :]
    
    # img0 = cv2.resize(img0, (target_w, target_h))
    img0 = lpips.im2tensor(img0).cuda()
    img1 = lpips.im2tensor(img1).cuda()

    # Compute distance
    dist01 = loss_fn.forward(img0, img1)
    return dist01.squeeze().detach().cpu().numpy()

def compute_mouth_lpips(pred_path, target_path, loss_fn, left, right, up, bottom):
    # Load images
    # img0 = lpips.im2tensor(lpips.load_image(pred_path)).cuda()
    # img1 = lpips.im2tensor(lpips.load_image(target_path)).cuda()
    
    img0 = lpips.load_image(pred_path)
    img1 = lpips.load_image(target_path)
    target_h, target_w = img1.shape[:2]
    
    img0 = cv2.resize(img0, (target_w, target_h))

    img0 = img0[up:bottom, left:right, :]
    img1 = img1[up:bottom, left:right, :]

    img0 = lpips.im2tensor(img0).cuda()
    img1 = lpips.im2tensor(img1).cuda()

    # Compute distance
    dist01 = loss_fn.forward(img0, img1)
    return dist01.detach().cpu().numpy()

def eval_image(args):
    sub_list = list()
    sub_list += ['renderonly_test_stage1_bookshelf_209999']
    sub_list += ['renderonly_test_stage1_path_209999']
    sub_list += ['renderonly_test_stage1_sofa1_209999']
    sub_list += ['renderonly_test_stage1_sofa2_209999']
    sub_list += ['renderonly_test_stage1_sofa3_209999']
    sub_list += ['renderonly_test_stage1_table_simple_209999']

    if 'lpips' in args.metric:
        loss_fn = lpips.LPIPS(net='alex',version='0.1',model_path='lpips_models/lpips_weights_v0.1/alex.pth').cuda()
        
    for sub_dir in sub_list:
        input_dir = os.path.join(args.input_dir, args.algo, sub_dir)
        # gt_input_dir = os.path.join(args.input_dir, args.algo, sub_dir)

        output_dir = os.path.join(args.output_dir, args.algo)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        input_file_list = os.listdir(input_dir)
        input_file_list = [input_file for input_file in input_file_list if '.jpg' in input_file]
        input_file_list = sorted(input_file_list)

        sum_metric = 0.0
        cnt = 0
        metric = args.metric
        
        for index, input_file in enumerate(tqdm(input_file_list)):
            pred_path = os.path.join(input_dir, input_file)
            # target_path = os.path.join(gt_input_dir, "{:05d}.jpg".format(int(input_file.replace('.jpg', ''))))
            
            if 'lpips' in metric:
                sum_metric += compute_lpips(pred_path, loss_fn)
            else:
                ori_image = cv2.imread(pred_path)
                H, W, _ = ori_image.shape 
                pred = ori_image[:, :W//4, :]
                target = ori_image[:, W//4:W//2, :]
                
                # target_h, target_w = target.shape[:2]
                # pred = cv2.resize(pred, (target_w, target_h))

                if 'psnr' in metric:
                    sum_metric += compute_psnr(pred, target)
                elif 'ssim' in metric:
                    sum_metric += compute_ssim(pred, target)
                
            cnt += 1
            
        sum_metric = sum_metric / cnt
        print(metric, '\t', sub_dir, '\t', sum_metric) 

        with open(os.path.join(output_dir, metric+'_'+sub_dir+'.txt'), 'w') as f:
            f.write(metric+'\t%.3f\n' % sum_metric) 

if __name__ == '__main__':
    args = parse_args()
    eval_image(args)
