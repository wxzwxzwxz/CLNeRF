import os
import sys
import lpips
import torch

loss_fn_alex = lpips.LPIPS(net='alex').cuda()
# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
img1 = torch.zeros(1,3,64,64)
d = loss_fn_alex(img0, img1)


def main():
    input_dir_1 = sys.argv[1]
    input_dir_2 = sys.argv[2]

    input_file_list_1 = get_img_list(input_dir_1)
    input_file_list_2 = get_img_list(input_dir_2)

    for idx in input_file_list_1:
        img_1 = read_image()
        img_2 = read_image()
        d = loss_fn_alex(img0, img1)
    