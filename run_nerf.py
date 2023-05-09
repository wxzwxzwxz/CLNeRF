import os, sys, cv2
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *
import nerf_teacher

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

from utils import x2samples
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk, return_feat=False):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        if return_feat == True:
            outputs = None
            outputs_dict = dict()

            for i in range(0, inputs.shape[0], chunk):
                output, output_dict = fn(inputs[i:i+chunk], return_feat=return_feat)
                # outputs_dict_list.append(output_dict)
                if outputs is None:
                    outputs = output
                    outputs_dict.update(output_dict)
                else:
                    outputs = torch.cat([outputs, output], 0)
                    for key in output_dict:
                        outputs_dict[key] = torch.cat([outputs_dict[key], output_dict[key]], 0)

            # return outputs, outputs_dict_list
            return outputs, outputs_dict
        else:
            return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0), None
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64, return_feat=False):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    if return_feat == True:
        outputs_flat, outputs_dict = batchify(fn, netchunk, return_feat=return_feat)(embedded)
        
        for key in outputs_dict:
            outputs_dict[key] = torch.reshape(outputs_dict[key], list(inputs.shape[:-1]) + [outputs_dict[key].shape[-1]])

        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs, outputs_dict
    else:
        outputs_flat, _ = batchify(fn, netchunk)(embedded)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs, None

def batchify_rays(rays_flat, chunk=1024*32, use_point_mask=False, 
                use_predict_mask=False,
                render_kwargs_test_teacher=None, 
                render_kwargs_test_teacher_second=None, 
                render_kwargs_test_mask=None, 
                **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs, use_point_mask=use_point_mask, 
                        use_predict_mask=use_predict_mask,
                        render_kwargs_test_teacher=render_kwargs_test_teacher, 
                        render_kwargs_test_teacher_second=render_kwargs_test_teacher_second,
                        render_kwargs_test_mask=render_kwargs_test_mask)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    
    # all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    for k in all_ret:
        if 'point_error' in k:
            pass
        else:
            all_ret[k] = torch.cat(all_ret[k], 0)
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  use_point_mask=False,
                  use_predict_mask=False,
                  render_kwargs_test_teacher=None,
                  render_kwargs_test_teacher_second=None,
                  render_kwargs_test_mask=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs, use_point_mask=use_point_mask, \
                            use_predict_mask=use_predict_mask, \
                            render_kwargs_test_teacher=render_kwargs_test_teacher, \
                            render_kwargs_test_teacher_second=render_kwargs_test_teacher_second, \
                            render_kwargs_test_mask=render_kwargs_test_mask)

    for k in all_ret:
        if 'point_error' in k:
            pass 
        else:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)
    
    # if use_mask_nerf == True:
    #     # rgb * mask, depth * mask, acc * mask
    #     all_ret_mask = batchify_rays(rays, chunk, **render_kwargs_test_mask)
    #     for k in all_ret_mask:
    #         k_sh = list(sh[:-1]) + list(all_ret_mask[k].shape[1:])
    #         all_ret_mask[k] = torch.reshape(all_ret_mask[k], k_sh)
    #         print(k, all_ret_mask[k].shape, all_ret[k].shape)

    #     print('use mask nerf')
    #     input()

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, args=None, \
                gt_imgs=None, savedir=None, render_factor=0, render_mask_only=False, output_paths=None, render_mask_threshold=0.1):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

        K[:2, :] = K[:2, :]/render_factor

    rgbs = []
    disps = []

    if args and args.use_predict_mask:
        masks = []

    t = time.time()
    psnr = 0
    for i, c2w in enumerate(tqdm(render_poses)):
    # for i, c2w in enumerate((render_poses)):
        # print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, extras = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], use_predict_mask=args.use_predict_mask, **render_kwargs)
        # for key in extras:
        #     print(key)
        # input()
        rgb = rgb.cpu().numpy()

        if gt_imgs is not None:
            gt_img = gt_imgs[i]
            gt_img = cv2.resize(gt_img, (W, H))
            
            error_map = np.abs(rgb-gt_img)
            error_mask = np.mean(error_map, 2) > render_mask_threshold # 0.1
            error_mask = np.stack([error_mask]*3, -1)

            # if error_mask.sum() > 2000:
            #     print(i)
            
            mse = img2mse_np(rgb, gt_img)
            # psnr += mse2psnr_np(mse)
            cur_psnr = mse2psnr_np(mse)
            psnr += cur_psnr

            if render_mask_only:
                rgb = error_mask
            else:
                rgb = np.concatenate([rgb, gt_img, error_map, error_mask], 1)
            
        rgbs.append(rgb)
        disps.append(disp.cpu().numpy())
        if args and args.use_predict_mask:
            masks.append(extras['mask_map'].cpu().numpy())

        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            if render_mask_only:
                # TODO: check!!!
                if output_paths is not None:
                    filename = os.path.join(savedir, output_paths[i].split('/')[-1])
                else:
                    filename = os.path.join(savedir, '{:04d}.png'.format(i))
                
                # cur_h, cur_w = rgb8.shape[:2]
                # rgb8 = cv2.resize(rgb8, (int(cur_w*2), int(cur_h*2)))
                imageio.imwrite(filename, rgb8)
            else:
                filename = os.path.join(savedir, '{:05d}.jpg'.format(i))
                cur_h, cur_w, _ = rgb8.shape
                # rgb8 =cv2.putText(img=np.copy(rgb8), text=str(cur_psnr), org=(cur_w-cur_w//4, cur_h-30), fontFace=3,  fontScale=2, color=(255,0,0), thickness=2)
                imageio.imwrite(filename, rgb8)
                
                disp = to8b(disps[-1])
                os.makedirs(savedir + '_disp', exist_ok=True)
                filename_disp = os.path.join(savedir + '_disp', '{:05d}.jpg'.format(i))
                imageio.imwrite(filename_disp, disp)

                if args and args.use_predict_mask:
                    mask = to8b(masks[-1])
                    os.makedirs(savedir + '_mask', exist_ok=True)
                    filename_mask = os.path.join(savedir + '_mask', '{:05d}.jpg'.format(i))
                    imageio.imwrite(filename_mask, mask)

    
    psnr = psnr / len(render_poses)
    print(psnr)
    
    if savedir is not None:
        with open(os.path.join(savedir, 'psnr.txt'), 'w') as f:
            f.write(str(psnr)+'\n')
    
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

def count_param(model):
    param_count = 0
    for param in model.parameters():
        if param.requires_grad == True:
            param_count += param.view(-1).size()[0]

    return param_count

# small func to check tuning params
def check_tuning_params(model_list):
    print("Tuning only:")
    for model in model_list:
        for n,v in model.named_parameters():
            if v.requires_grad ==True:
                print(n)

def ft(model,ft_layers):
    #selective FT
    for n,v in model.named_parameters():
        if any([l in n for l in ft_layers]):
            v.requires_grad_(True)

def bitfit(model,keeps,bitfit_layers):

    # selective Bitfit
    for n,v in model.named_parameters():
        if (keeps in n) and any([l in n for l in bitfit_layers]):
            v.requires_grad_(True)

def set_grad_false_except_keyword(model, model_fine, keyword_list):
    # print(keyword_list)
    for n,v in model.named_parameters():
        v.requires_grad_(False)

    for n,v in model.named_parameters():
        if any([l in n for l in keyword_list]):
            v.requires_grad_(True)
    
    if model_fine is not None:
        for n,v in model_fine.named_parameters():
            v.requires_grad_(False)

        for n,v in model_fine.named_parameters():
            if any([l in n for l in keyword_list]):
                v.requires_grad_(True)

def set_grad_true_in_keyword(model, model_fine, keyword_list):
    for n,v in model.named_parameters():
        if any([l in n for l in keyword_list]):
            v.requires_grad_(True)

    if model_fine is not None:
        for n,v in model_fine.named_parameters():
            if any([l in n for l in keyword_list]):
                v.requires_grad_(True)

def create_nerf(args, ckpt_path=None):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, args=args).to(device)

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, args=args).to(device)
    
    # partly finetune
    # only finetune adaptor
    if args.lora:
        print('usingh LoRA')
        lora.mark_only_lora_as_trainable(model)
        lora.mark_only_lora_as_trainable(model_fine)
        print("Tuning only:")

        for n,v in model.named_parameters():
            if v.requires_grad ==True:
                print(n)
        if model_fine is not None:
            for n,v in model_fine.named_parameters():
                if v.requires_grad ==True:
                    print(n)
    elif args.adapter_layers:
        keyword_list = ['adapters']
        # keyword_list += ['feature_linear', 'alpha_linear', 'rgb_linear', 'pts_linears.7'] # v2
        # keyword_list += ['feature_linear', 'alpha_linear', 'rgb_linear'] # v3
        # keyword_list += ['alpha_linear'] # v4
        
        set_grad_false_except_keyword(model, model_fine, keyword_list)
    elif args.use_expert:
        keyword_list = ['expert']
        if args.use_expert_ft_alpha:
            keyword_list += ['alpha_linear']
        if args.use_expert_ft_rgbfeat:
            keyword_list += ['rgb_linear']
            keyword_list += ['output_linear']
        if args.use_predict_mask:
            keyword_list += ['mask_linear']

        set_grad_false_except_keyword(model, model_fine, keyword_list)
    elif args.bitfit:
        # freeze model params except bias
        _ignores=[]
        _keeps='bias'
        ft_layers=args.ft_layers #['pts_linears.7','alpha_linear','rgb_linear','feature_linear','views_linears']
        bitfit_layers=args.bitfit_layers #['pts_linears.0','pts_linears.1','pts_linears.2','pts_linears.3',
                       #'pts_linears.4','pts_linears.5','pts_linears.6']

        #freeze all
        for n,v in model.named_parameters():
            v.requires_grad_(False)
        
        bitfit(model,_keeps,bitfit_layers) # bitfit for model
        if ft_layers is not None:
            ft(model,ft_layers) 
        
        if model_fine is not None:
            # model fine
            #freeze all
            for n,v in model_fine.named_parameters():
                v.requires_grad_(False)
            
            # selective Bitfit
            bitfit(model_fine,_keeps,bitfit_layers)
            
            #selective FT
            if ft_layers is not None:
                ft(model_fine,ft_layers) 
    


    # finetune several layers
    if args.finetune_last_layer_only == True:
        keyword_list = ['alpha_linear', 'rgb_linear', 'views_linears']
        set_grad_true_in_keyword(model, model_fine, keyword_list)

    elif args.finetune_last_layers > 0:
        keyword_list = ['alpha_linear', 'rgb_linear', 'views_linears']
        set_grad_true_in_keyword(model, model_fine, keyword_list)
        for z in model.named_parameters():
            if 'pts_linears' in z[0] and int(z[0].split('.')[1]) > 7 - args.finetune_last_layers:
                set_grad_true_in_keyword(model, model_fine, [z[0]])
            
    if not args.render_only:
        check_tuning_params([model, model_fine])

    # Create optimizer
    if args.adapter_layers:
        # grad_vars = list(model.parameters())
        grad_vars = [z[1] for z in model.named_parameters() if 'adapters' not in z[0] and z[1].requires_grad == True]
        if args.N_importance > 0: 
            # grad_vars += list(model_fine.parameters())
            grad_vars += [z[1] for z in model_fine.named_parameters() if 'adapters' not in z[0] and z[1].requires_grad == True]
        
        if len(grad_vars) != 0:
            optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
        else:
            optimizer = None

        grad_vars_second = [z[1] for z in model.named_parameters() if 'adapters' in z[0] and z[1].requires_grad == True]
        if args.N_importance > 0: 
            # grad_vars += list(model_fine.parameters())
            grad_vars_second += [z[1] for z in model_fine.named_parameters() if 'adapters' in z[0] and z[1].requires_grad == True]
        optimizer_second = torch.optim.Adam(params=grad_vars_second, lr=args.lrate_adaptor, betas=(0.9, 0.999))
    elif args.use_expert:
        grad_vars = [z[1] for z in model.named_parameters() if 'expert' not in z[0] and z[1].requires_grad == True]
        if args.N_importance > 0: 
            grad_vars += [z[1] for z in model_fine.named_parameters() if 'expert' not in z[0] and z[1].requires_grad == True]
        
        if len(grad_vars) != 0:
            optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
        else:
            optimizer = None

        grad_vars_second = [z[1] for z in model.named_parameters() if 'expert' in z[0] and z[1].requires_grad == True]
        if args.N_importance > 0: 
            grad_vars_second += [z[1] for z in model_fine.named_parameters() if 'expert' in z[0] and z[1].requires_grad == True]
        optimizer_second = torch.optim.Adam(params=grad_vars_second, lr=args.lrate_adaptor, betas=(0.9, 0.999))
    else:
        # grad_vars = list(model.parameters())
        grad_vars = [z[1] for z in model.named_parameters() if 'emb_linear' not in z[0] and z[1].requires_grad == True]
        if args.N_importance > 0: 
            # grad_vars += list(model_fine.parameters())
            grad_vars += [z[1] for z in model_fine.named_parameters() if 'emb_linear' not in z[0] and z[1].requires_grad == True]
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
        optimizer_second = None

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if ckpt_path is not None:
        ckpts = [ckpt_path]
    elif args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f and 'adaptor.tar' not in f]

    # print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        try:
            if optimizer is not None:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            # if args.adapter_layers:
            #     optimizer_second.load_state_dict(ckpt['optimizer_state_dict'])
            # elif args.use_expert:
            #     optimizer_second.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception as e:
            pass
            # print(e)

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'], strict=False)
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'], strict=False)

        if args.adapter_layers:
            try:
                ckpt_adaptor = torch.load(ckpt_path.replace('.tar', '_adaptor.tar'))
                model.adapters.load_state_dict(ckpt_adaptor['network_fn_adapters_state_dict'], strict=False)
                if model_fine is not None:
                    model_fine.adapters.load_state_dict(ckpt_adaptor['network_fine_adapters_state_dict'], strict=False)
            except Exception as e:
                print(e)
        elif args.use_expert:
            try:
                ckpt_expert = torch.load(ckpt_path.replace('.tar', '_expert.tar'))
                model.expert.load_state_dict(ckpt_expert['network_fn_expert_state_dict'], strict=False)
                if model_fine is not None:
                    model_fine.expert.load_state_dict(ckpt_expert['network_fine_expert_state_dict'], strict=False)
            except Exception as e:
                print(e)

    if args.add_dino:
        assert args.ft_path is not None, 'for now use only pre-trained model'
        # add emb parameters after loading optimizer, freeze for 1000 iter
        utils.initialize_optimizer(optimizer, model, model_fine)
        utils.set_requires_grad(model, keys_excl=['emb_linear', 'emb_linear_penultimate'], requires_grad=False)
        utils.set_requires_grad(model_fine, keys_excl=['emb_linear', 'emb_linear_penultimate'], requires_grad=False)


    # compute params
    param_count = count_param(model)
    param_count += count_param(model_fine)
    print('Total params:', param_count)

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk,
                                                                return_feat=args.return_feat)
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, optimizer_second

def create_teacher_nerf(args, ckpt_path=None):
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = nerf_teacher.NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, args=args).to(device)

    model_fine = None
    if args.N_importance > 0:
        model_fine = nerf_teacher.NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, args=args).to(device)

    set_grad_false_except_keyword(model, model_fine, [])

    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if ckpt_path is not None:
        ckpts = [ckpt_path]
    elif args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f and 'adaptor.tar' not in f]

    # print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'], strict=False)
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'], strict=False)

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk,
                                                                return_feat=args.return_feat)
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test

def create_mask_nerf(args, ckpt_path):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    
    # output_ch = 5 if args.N_importance > 0 else 4
    output_ch = 4
    skips = [4]
    # model = NeRF(D=args.netdepth, W=args.netwidth,
    #              input_ch=input_ch, output_ch=output_ch, skips=skips,
    #              input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    model = NeRF(D=args.netdepth_mask, W=args.netwidth_mask,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, args=args).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    # if args.N_importance > 0:
    #     model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
    #                       input_ch=input_ch, output_ch=output_ch, skips=skips,
    #                       input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    #     grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    # optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    # start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    ckpts = [ckpt_path]
    # if ckpt_path is not None:
    #     ckpts = [ckpt_path]
    # elif args.ft_path is not None and args.ft_path!='None':
    #     ckpts = [args.ft_path]
    # else:
    #     ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if '_mask.tar' in f]

    # print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        # start = ckpt['global_step']
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'], strict=False)
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'], strict=False)

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : 0, # args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'use_viewdirs' : args.use_viewdirs,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    # return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer
    return render_kwargs_test

def raw2weights(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    # emb = torch.tanh(raw[...,-64:])  # [N_rays, N_samples, 3]

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    # rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    
    # depth_map = torch.sum(weights * z_vals, -1)
    # disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    # acc_map = torch.sum(weights, -1)

    # if white_bkgd:
    #     rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb, alpha, weights

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False,
                use_predict_mask=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    if use_predict_mask:
        mask = torch.sigmoid(raw[...,-1:])
        mask_map = torch.sum(weights[...,None] * mask, -2)  # [N_rays, 3]
    else:
        mask_map = None 

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])
        if use_predict_mask:
            mask_map = mask_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map, mask_map

def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                use_point_mask=False,
                point_mask_threshold=0.9,
                use_predict_mask=False,
                render_kwargs_test_teacher=None,
                render_kwargs_test_teacher_second=None, 
                render_kwargs_test_mask=None,
                stop_pdf_sampling=False,
                ):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    # print(pts[:, :, 0].min(), pts[:, :, 0].max()) # scene: -1, -3, 2, lego: -1, 1 -1, 1 -2, 2
    # print(pts[:, :, 1].min(), pts[:, :, 1].max()) # scene: -1, -3, 2, lego: -1, 1 -1, 1 -2, 2
    # print(pts[:, :, 2].min(), pts[:, :, 2].max()) # scene: -1, -3, 2, lego: -1, 1 -1, 1 -2, 2
    # input()

#     raw = run_network(pts)
    raw, feat_dict = network_query_fn(pts, viewdirs, network_fn)
    if use_point_mask:
        with torch.no_grad():
            raw_teacher, feat_dict_teacher = network_query_fn(pts, viewdirs, render_kwargs_test_teacher['network_fn'])
            raw_mask, _ = network_query_fn(pts, viewdirs, render_kwargs_test_mask['network_fn'])
            
            if render_kwargs_test_teacher_second is not None:
                raw_teacher_second, feat_dict_teacher_second = network_query_fn(pts, viewdirs, render_kwargs_test_teacher_second['network_fn'])

            # generate mask
            point_rgb, point_alpha, weights = raw2weights(raw_mask, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
            mask = point_rgb * point_alpha.unsqueeze(-1)
            
            mask = torch.mean(mask, -1)
            mask = mask > point_mask_threshold
            mask = mask.int()
            
            mask = mask.unsqueeze(-1)
        
        # for key in feat_dict:
        #     torch.mean(torch.abs(feat_dict[key] - feat_dict_teacher[key]))

        point_error = torch.abs(raw_teacher * (1-mask) - raw * (1-mask))
        point_error = torch.sum(point_error, 1) / (torch.sum((1-mask), 1)+1e-6)

        if render_kwargs_test_teacher_second is not None:
            point_error_second = torch.abs(raw_teacher_second * (1-mask) - raw * (1-mask))
            point_error_second = torch.sum(point_error_second, 1) / (torch.sum((1-mask), 1)+1e-6)

    rgb_map, disp_map, acc_map, weights, depth_map, mask_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, use_predict_mask=use_predict_mask)
    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        if use_predict_mask:
            mask_map_0 = mask_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        
        if stop_pdf_sampling:
            z_samples = sample_pdf(z_vals_mid, torch.ones_like(weights[...,1:-1]), N_importance, det=(perturb==0.), pytest=pytest)
        else:
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)

        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw, feat_dict = network_query_fn(pts, viewdirs, run_fn)
        
        if use_point_mask:
            point_error_0 = point_error
            if render_kwargs_test_teacher_second is not None:
                point_error_0_second = point_error_second
                
            with torch.no_grad():
                raw_teacher, _ = network_query_fn(pts, viewdirs, render_kwargs_test_teacher['network_fine'])
                raw_mask, _ = network_query_fn(pts, viewdirs, render_kwargs_test_mask['network_fn'])
                
                if render_kwargs_test_teacher_second is not None:
                    raw_teacher_second, _ = network_query_fn(pts, viewdirs, render_kwargs_test_teacher_second['network_fn'])

                # generate mask
                point_rgb, point_alpha, weights = raw2weights(raw_mask, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
                mask = point_rgb * point_alpha.unsqueeze(-1)
                mask = torch.mean(mask, -1)
                mask = mask > point_mask_threshold
                mask = mask.int()
                mask = mask.unsqueeze(-1)
            
            point_error = torch.abs(raw_teacher * (1-mask) - raw * (1-mask))
            # point_error = torch.sum(point_error) / (torch.sum(1-mask)+1e-6)
            point_error = torch.sum(point_error, 1) / (torch.sum((1-mask), 1)+1e-6)

            if render_kwargs_test_teacher_second is not None:
                point_error_second = torch.abs(raw_teacher_second * (1-mask) - raw * (1-mask))
                # point_error_second = torch.sum(point_error_second) / (torch.sum(1-mask)+1e-6)
                point_error_second = torch.sum(point_error_second, 1) / (torch.sum((1-mask), 1)+1e-6)

        rgb_map, disp_map, acc_map, weights, depth_map, mask_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, use_predict_mask=use_predict_mask)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if use_predict_mask:
        ret['mask_map'] = mask_map

    if retraw:
        ret['raw'] = raw

    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        if use_predict_mask:
            ret['mask_map0'] = mask_map_0
    
    if use_point_mask:
        ret['point_error'] = point_error
        if N_importance > 0:
            ret['point_error0'] = point_error_0
        
        if render_kwargs_test_teacher_second is not None:
            ret['point_error_second'] = point_error_second
            if N_importance > 0:
                ret['point_error0_second'] = point_error_0_second

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--netdepth_mask", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth_mask", type=int, default=128, 
                        help='channels per layer')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_mask_only", action='store_true')
    parser.add_argument("--render_mask_threshold", type=float, default=0.1)
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=200000, 
                        help='frequency of render_poses video saving')
    
    # experiments
    parser.add_argument("--near", type=float, default=None)
    parser.add_argument("--far", type=float, default=None)
    parser.add_argument("--N_iters", type=int, default=200000)
    parser.add_argument("--scene_scale", type=float, default=None)
    parser.add_argument("--use_teacher_nerf", action='store_true')
    parser.add_argument("--use_teacher_nerf_second", action='store_true')
    parser.add_argument("--use_point_mask", action='store_true')
    parser.add_argument("--datadir_teacher", type=str, default='./data/llff/fern', 
                        help='input data directory')
    parser.add_argument("--ft_teacher_path", type=str, default=None, 
                        help='specific weights npy file to reload for teacher network')
    parser.add_argument("--ft_mask_path", type=str, default=None, 
                        help='specific weights npy file to reload for mask network')
    parser.add_argument("--N_rand_teacher", type=int, default=1024, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--render_wo_images", action='store_true')
    parser.add_argument("--ori_H", type=float, default=None)
    parser.add_argument("--ori_W", type=float, default=None)
    parser.add_argument("--w_loss_teacher", type=float, default=0.1)
    parser.add_argument("--ext", type=str, default='.png')
    # parser.add_argument("--transforms_train", type=str, default=None)
    parser.add_argument('--transforms_train', nargs='+', default=None)
    parser.add_argument('--transforms_train_ratio', nargs='+', default=None)
    parser.add_argument("--transforms_val", type=str, default=None)
    # parser.add_argument("--transforms_test", type=str, default=None)
    parser.add_argument('--transforms_test', nargs='+', default=None)
    parser.add_argument('--transforms_test_ratio', nargs='+', default=None)
    parser.add_argument("--trainskip", type=int, default=1)
    parser.add_argument("--add_dino", type=bool, default=False)
    parser.add_argument("--dino_dir", type=str, default='')
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--point_mask_threshold", type=float, default=0.9)
    parser.add_argument("--finetune_last_layers", type=int, default=0)
    parser.add_argument("--finetune_last_layer_only", type=bool, default=False)
    parser.add_argument("--spherical_radius", type=float, default=4.0)
    parser.add_argument("--bitfit", action='store_true')
    parser.add_argument("--lora", action='store_true')
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument('--ft_layers', nargs='+', help='<Required> Set flag')
    parser.add_argument('--bitfit_layers', nargs='+', help='<Required> Set flag')
    parser.add_argument('--lora_layers', nargs='+', help='<Required> Set flag', default=None)
    
    parser.add_argument("--render_output_name", type=str, default=None)
    parser.add_argument("--transforms_test_key", type=str, default=None)
    parser.add_argument("--return_feat", type=bool, default=False)
    parser.add_argument("--use_mask_bce_loss", type=bool, default=False)
    parser.add_argument("--use_mask_reg_loss", type=bool, default=False)
    parser.add_argument("--w_mask_reg_loss", type=float, default=1e-3)
    parser.add_argument("--use_lr_global_step_from_scratch", type=bool, default=False)

    # for adapter
    parser.add_argument('--adapter_layers', nargs='+', help='[0-6]. 0 indicate after input layer,pts 7 do not support adapter', default=None)
    parser.add_argument('--adapter_version', type=int, default=0)
    parser.add_argument('--bottle_neck_dim', type=int, default=None)
    parser.add_argument("--lrate_adaptor", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay_adaptor", type=int, default=500, help='exponential learning rate decay (in 1000 steps)')

    # for expert
    parser.add_argument("--use_expert", type=bool, default=False)
    parser.add_argument("--use_expert_ft_alpha", type=bool, default=False)
    parser.add_argument("--use_expert_ft_rgbfeat", type=bool, default=False)
    parser.add_argument("--expert_version", type=str, default='v2')
    parser.add_argument("--expert_w", type=int, default=256)
    parser.add_argument("--expert_d", type=int, default=2)
    parser.add_argument("--use_predict_mask", type=bool, default=False)

    parser.add_argument("--load_mask", type=bool, default=False)
    parser.add_argument("--load_mask_dir", type=str, default=None)
    parser.add_argument("--mask_ext", type=str, default=None)
    parser.add_argument("--w_mask_loss", type=float, default=0.1)
    
    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        if args.render_wo_images:
            poses, render_poses, hwf, i_split, output_paths, _, _ = load_blender_data(args, args.datadir, args.half_res, args.testskip, 
                                                                                load_imgs=False, ori_H=args.ori_H, ori_W=args.ori_W, ext=args.ext,
                                                                                transforms_train=args.transforms_train, transforms_val=args.transforms_val, transforms_test=args.transforms_test,
                                                                                trainskip=args.trainskip, spherical_radius=args.spherical_radius,
                                                                                transforms_train_ratio=args.transforms_train_ratio,
                                                                                transforms_test_ratio=args.transforms_test_ratio,)
        else:
            images, poses, render_poses, hwf, i_split, output_paths, ori_H, ori_W, fts_train, fts_test, images_mask = load_blender_data(args, args.datadir, args.half_res, args.testskip, ext=args.ext,
                                                                                                        transforms_train=args.transforms_train, transforms_val=args.transforms_val, transforms_test=args.transforms_test,
                                                                                                        trainskip=args.trainskip, spherical_radius=args.spherical_radius,
                                                                                                        transforms_train_ratio=args.transforms_train_ratio,
                                                                                                        transforms_test_ratio=args.transforms_test_ratio)
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
                if args.load_mask:
                    images_mask = images_mask[...,:3]*images_mask[...,-1:] + (1.-images_mask[...,-1:])
            else:
                images = images[...,:3]
                if args.load_mask:
                    images_mask = images_mask[..., :3]
                
        print('Loaded blender', poses.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        if args.near:
            near = args.near
            far = args.far
        else:
            near = 2.
            far = 6.
            
        if args.use_teacher_nerf:
            poses_teacher, render_poses_teacher, _, i_split_teacher, _, fts_train, fts_test = load_blender_data(args, args.datadir_teacher, args.half_res, args.testskip, load_imgs=False, ori_H=ori_H, ori_W=ori_W, ext=args.ext,
                                                                                            transforms_train=args.transforms_train, transforms_val=args.transforms_val, transforms_test=args.transforms_test,
                                                                                            trainskip=args.trainskip, spherical_radius=args.spherical_radius,
                                                                                            transforms_train_ratio=args.transforms_train_ratio,
                                                                                            transforms_test_ratio=args.transforms_test_ratio)
            print('Loaded blender for teacher', poses_teacher.shape, render_poses_teacher.shape, args.datadir_teacher)
            i_train_teacher, i_val_teacher, i_test_teacher = i_split_teacher

            # if args.near:
            #     near = args.near
            #     far = args.far
            # else:
            #     near = 2.
            #     far = 6.
                
            # if args.white_bkgd:
            #     images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            # else:
            #     images = images[...,:3]

        if args.use_teacher_nerf_second:
            poses_teacher_second, render_poses_teacher_second, _, i_split_teacher_second, _, fts_train, fts_test, _ = load_blender_data(args, args.datadir_teacher_second, args.half_res, args.testskip, load_imgs=False, ori_H=ori_H, ori_W=ori_W,
                                                                                                                transforms_train=args.transforms_train, transforms_val=args.transforms_val, transforms_test=args.transforms_test,
                                                                                                                trainskip=args.trainskip, spherical_radius=args.spherical_radius,
                                                                                                                transforms_train_ratio=args.transforms_train_ratio,
                                                                                                                transforms_test_ratio=args.transforms_test_ratio)
            print('Loaded blender for second teacher', poses_teacher_second.shape, render_poses_teacher_second.shape, args.datadir_teacher_second)
            i_train_teacher_second, _, _ = i_split_teacher_second

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])
        output_paths = output_paths[i_test]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    
    if args.config is not None:   
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    if args.adapter_layers:
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, optimizer_second = create_nerf(args, ckpt_path=args.ckpt_path)
    elif args.use_expert:
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, optimizer_second = create_nerf(args, ckpt_path=args.ckpt_path)
    else:
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, _ = create_nerf(args, ckpt_path=args.ckpt_path)

    global_step = start
    lr_global_step = 0
    if args.adapter_layers:
        global_step_second = 0
    elif args.use_expert:
        global_step_second = 0

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # update point-mask nerf teacher model
    if args.use_point_mask:
        bds_dict = {
            'point_mask_threshold' : args.point_mask_threshold,
        }
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)

    # Create teacher nerf model
    if args.use_teacher_nerf:
        # _, render_kwargs_test_teacher, _, _, _, _ = create_nerf(args, ckpt_path=args.ft_teacher_path)
        _, render_kwargs_test_teacher = create_teacher_nerf(args, ckpt_path=args.ft_teacher_path)
        
        render_kwargs_test_teacher.update(bds_dict)
        render_kwargs_test_mask = create_mask_nerf(args, ckpt_path=args.ft_mask_path)
        render_kwargs_test_mask.update(bds_dict)
    else:
        render_kwargs_test_teacher = None
        render_kwargs_test_mask = None

    if args.use_teacher_nerf_second:
        # _, render_kwargs_test_teacher_second, _, _, _, _ = create_nerf(args, ckpt_path=args.ft_teacher_path_second)
        _, render_kwargs_test_teacher_second = create_teacher_nerf(args, ckpt_path=args.ft_teacher_path_second)
        render_kwargs_test_teacher_second.update(bds_dict)
    else:
        render_kwargs_test_teacher_second = None
    
    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_wo_images:
                images = None
            elif args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            if args.render_output_name is not None:
                testsavedir = os.path.join(basedir, expname, args.render_output_name + '_{:06d}'.format(start))
                if args.render_factor > 0:
                    testsavedir = os.path.join(basedir, expname, args.render_output_name + '_{:06d}'.format(start) + '_f' + str(args.render_factor))
            else:
                testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))

                if args.transforms_test is not None:
                    if isinstance(args.transforms_test, list):
                        testsavedir = testsavedir + '_' + args.transforms_test[0].split('.')[0].split('_')[-1]
                    else:
                        testsavedir = testsavedir + '_' + args.transforms_test.split('.')[0].split('_')[-1]

            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, args=args, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor, render_mask_only=args.render_mask_only, output_paths=output_paths, render_mask_threshold=args.render_mask_threshold)
            print('Done rendering', testsavedir)
            # imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)
        
        if args.add_dino:
            # rays_emb = x2samples(features, i_train)
            rays_emb = x2samples_new(fts_train)
            rays_emb = rays_emb[rand_idx]

        print('done')
        i_batch = 0
    
    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)
        if args.add_dino:
            rays_emb = torch.Tensor(rays_emb).to(device)

    # N_iters = 200000 + 1
    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]
            
            if args.add_dino:
                batch_emb = rays_emb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
                batch_emb = torch.transpose(batch_emb, 0, 1)
                target_emb = batch_emb[0]
                # assert target_emb.shape[0] == N_rand
                if target_emb.shape[0] != N_rand:
                    print('UNEQUAL')
                    print(target_emb.shape[0], N_rand)
                assert target_emb.shape[1] == 64

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                if args.add_dino:
                    rays_emb = rays_emb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            if args.load_mask:
                target_mask = images_mask[img_i]
                target_mask = torch.Tensor(target_mask).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                if args.load_mask:
                    target_s_mask = target_mask[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                use_predict_mask=args.use_predict_mask,
                                                stop_pdf_sampling= i - start < 0,
                                                **render_kwargs_train)
    
        if optimizer is not None:
            optimizer.zero_grad()
        if args.adapter_layers:
            optimizer_second.zero_grad()
        elif args.use_expert:
            optimizer_second.zero_grad()

        if args.use_mask_bce_loss:
            img_loss = img2bce(rgb, target_s)
            try:
                psnr = mse2psnr(img2mse(rgb, target_s))
            except:
                psnr = 0
        else:
            # if args.use_predict_mask:
            #     img_loss = img2mse_withmask(rgb, target_s, extras['mask_map'])
            #     psnr = mse2psnr(img_loss)
            # else:
            img_loss = img2mse(rgb, target_s)
            psnr = mse2psnr(img_loss)
        
        if args.use_mask_reg_loss:
            img_loss += torch.mean(torch.abs((1 - rgb))) * args.w_mask_reg_loss

        trans = extras['raw'][...,-1]
        loss = img_loss
        
        if args.add_dino:
            distances = ((emb - target_emb) ** 2).sum(dim=1)
            loss_distillation = distances.mean() * 0.001
            # print('rgb vs. ft', loss, loss_distillation, loss / loss_distillation)
            loss = loss + loss_distillation

        if 'rgb0' in extras:
            if args.use_mask_bce_loss:
                img_loss0 = img2bce(extras['rgb0'], target_s)
                try:
                    psnr0 = mse2psnr(img2mse(rgb, target_s))
                except:
                    psnr0 = 0
            else:
                # if args.use_predict_mask:
                #     img_loss0 = img2mse_withmask(extras['rgb0'], target_s, extras['mask_map0'])
                #     psnr0 = mse2psnr(img_loss0)
                # else:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                psnr0 = mse2psnr(img_loss0)
            
            if args.use_mask_reg_loss:
                img_loss0 += torch.mean(torch.abs((1 - extras['rgb0']))) * args.w_mask_reg_loss

            loss = loss + img_loss0
                
            if args.add_dino:
                distances = ((extras['emb0'] - target_emb) ** 2).sum(dim=1)
                loss_distillation0 = distances.mean() * 0.001
                # print('rgb vs. ft', img_loss0, loss_distillation0, img_loss0 / loss_distillation0)
                loss = loss + loss_distillation0
        
        if args.use_predict_mask:
            loss_mask = img2mse(extras['mask_map'], target_s_mask) # torch.mean(torch.abs((1 - extras['mask_map']))) # * args.w_mask_reg_loss
            if 'rgb0' in extras:
                loss_mask += img2mse(extras['mask_map0'], target_s_mask)
            
            loss = loss + args.w_mask_loss * loss_mask

        if args.use_teacher_nerf:
            # randomly sample rays and genrate ground truth
            batch_rays = sample_rays(args, hwf, K, i, i_train_teacher, args.N_rand_teacher, poses_teacher, start)
            
            if args.use_predict_mask:
                rgb_student, disp_student, acc_student, extras_student = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                            verbose=i < 10, retraw=True,
                                                            use_predict_mask=args.use_predict_mask,
                                                            stop_pdf_sampling= i - start < 0,
                                                            **render_kwargs_train)
                                                        
                with torch.no_grad():
                    rgb_teacher, disp_teacher, acc_teacher, _ = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                            verbose=i < 10, retraw=True,
                                                            stop_pdf_sampling= i - start < 0,
                                                            **render_kwargs_test_teacher)

                    # rgb_gt = (1-rgb_mask) * rgb_teacher

                img_loss_teacher = img2mse_withmask(rgb_student, rgb_teacher, (1-extras['mask_map']))
                
                loss += img_loss_teacher * args.w_loss_teacher

                if 'rgb0' in extras:
                    img_loss0_teacher = img2mse_withmask(extras_student['rgb0'], rgb_teacher, (1-extras['mask_map']))
                    loss += img_loss0_teacher * args.w_loss_teacher

            elif args.use_point_mask:
                rgb_student, disp_student, acc_student, extras_student = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                        verbose=i < 10, retraw=True,
                                                        use_point_mask=args.use_point_mask,
                                                        render_kwargs_test_teacher=render_kwargs_test_teacher,
                                                        render_kwargs_test_teacher_second=render_kwargs_test_teacher_second,
                                                        render_kwargs_test_mask=render_kwargs_test_mask,
                                                        stop_pdf_sampling= i - start < 0,
                                                        **render_kwargs_train)
                # loss_teacher = extras_student['point_error'][0] + extras_student['point_error0'][0]
                # loss += loss_teacher * args.w_loss_teacher
                with torch.no_grad():
                    rgb_mask, disp_mask, acc_mask, _ = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                            verbose=i < 10, retraw=True,
                                                            stop_pdf_sampling= i - start < 0,
                                                            **render_kwargs_test_mask)
                rgb_mask = torch.mean(rgb_mask, -1).unsqueeze(-1)
                loss_teacher = torch.sum(extras_student['point_error'][0] * rgb_mask) / (torch.sum(rgb_mask) + 1e-6)
                loss_teacher += torch.sum(extras_student['point_error0'][0] * rgb_mask) / (torch.sum(rgb_mask) + 1e-6)
                loss += loss_teacher * args.w_loss_teacher

                if args.use_teacher_nerf_second:
                    pass
                    loss_teacher_second = extras_student['point_error_second'][0] + extras_student['point_error0_second'][0]
                    loss += loss_teacher_second * args.w_loss_teacher_second
            else:
                rgb_student, disp_student, acc_student, extras_student = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                            verbose=i < 10, retraw=True,
                                                            stop_pdf_sampling= i - start < 0,
                                                            **render_kwargs_train)
                                                        
                with torch.no_grad():
                    rgb_teacher, disp_teacher, acc_teacher, _ = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                            verbose=i < 10, retraw=True,
                                                            stop_pdf_sampling= i - start < 0,
                                                            **render_kwargs_test_teacher)
                    rgb_mask, disp_mask, acc_mask, _ = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                            verbose=i < 10, retraw=True,
                                                            stop_pdf_sampling= i - start < 0,
                                                            **render_kwargs_test_mask)
                    # rgb_gt = (1-rgb_mask) * rgb_teacher

                # img_loss_teacher = img2mse((1-rgb_mask) * rgb_student, rgb_gt)
                img_loss_teacher = img2mse_withmask(rgb_student, rgb_teacher, (1-rgb_mask))
                
                loss += img_loss_teacher * args.w_loss_teacher

                if 'rgb0' in extras:
                    # img_loss0_teacher = img2mse((1-rgb_mask) * extras_student['rgb0'], rgb_gt)
                    img_loss0_teacher = img2mse_withmask(extras_student['rgb0'], rgb_teacher, (1-rgb_mask))
                    loss += img_loss0_teacher * args.w_loss_teacher

        loss.backward()
        if optimizer is not None:
            optimizer.step()

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            if args.use_lr_global_step_from_scratch:
                decay_rate = 0.1
                decay_steps = args.lrate_decay * 1000
                new_lrate = args.lrate * (decay_rate ** (lr_global_step / decay_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate
                lr_global_step += 1
            else:
                decay_rate = 0.1
                decay_steps = args.lrate_decay * 1000
                new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate
        
        if args.adapter_layers:
            optimizer_second.step()

            decay_rate = 0.1
            decay_steps = args.lrate_decay_adaptor * 1000
            new_lrate = args.lrate_adaptor * (decay_rate ** (global_step_second / decay_steps))
            for param_group in optimizer_second.param_groups:
                param_group['lr'] = new_lrate 
            global_step_second += 1
        elif args.use_expert:
            optimizer_second.step()

            decay_rate = 0.1
            decay_steps = args.lrate_decay_adaptor * 1000
            new_lrate = args.lrate_adaptor * (decay_rate ** (global_step_second / decay_steps))
            for param_group in optimizer_second.param_groups:
                param_group['lr'] = new_lrate 
            global_step_second += 1

        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            output_dict = dict()
            
            '''
            model_dict = model.state_dict()
            modelCheckpoint = torch.load(checkpoint)
            pretrained_dict = modelCheckpoint['state_dict']

            network_fn_state_dict = render_kwargs_train['network_fn'].state_dict()
            
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            '''
            # for key in render_kwargs_train['network_fn'].state_dict().keys():
            #     print(key)
            # input()
            output_dict = {
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict()
            }
            
            if 'network_fine' in render_kwargs_train and render_kwargs_train['network_fine'] is not None:
                output_dict['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()
            
            if optimizer is not None:
                output_dict['optimizer_state_dict'] = optimizer.state_dict()

            if args.adapter_layers:
                path_second = os.path.join(basedir, expname, '{:06d}_adaptor.tar'.format(i))
                if 'network_fine' in render_kwargs_train and render_kwargs_train['network_fine'] is not None:
                    torch.save({
                        'global_step': global_step,
                        'network_fn_adapters_state_dict': render_kwargs_train['network_fn'].adapters.state_dict(),
                        'network_fine_adapters_state_dict': render_kwargs_train['network_fine'].adapters.state_dict(),
                    }, path_second)
                else:
                    torch.save({
                        'global_step': global_step,
                        'network_fn_adapters_state_dict': render_kwargs_train['network_fn'].adapters.state_dict(),
                    }, path_second)
            elif args.use_expert:
                # delete expert in original ckpt
                network_fn_state_dict = render_kwargs_train['network_fn'].state_dict()
                new_network_fn_state_dict = {k: v for k, v in network_fn_state_dict.items() if 'expert' not in k}
                output_dict['network_fn_state_dict'] = new_network_fn_state_dict

                # save expert in new ckpt
                path_second = os.path.join(basedir, expname, '{:06d}_expert.tar'.format(i))
                output_dict_expert = {
                    'global_step': global_step,
                    'network_fn_expert_state_dict': render_kwargs_train['network_fn'].expert.state_dict()
                }
                if 'network_fine' in render_kwargs_train and render_kwargs_train['network_fine'] is not None:
                    # torch.save({
                    #     'global_step': global_step,
                    #     'network_fn_expert_state_dict': render_kwargs_train['network_fn'].expert.state_dict(),
                    #     'network_fine_expert_state_dict': render_kwargs_train['network_fine'].expert.state_dict(),
                    # }, path_second)

                    # delete expert in original ckpt
                    network_fn_state_dict = render_kwargs_train['network_fine'].state_dict()
                    new_network_fn_state_dict = {k: v for k, v in network_fn_state_dict.items() if 'expert' not in k}
                    output_dict['network_fine_state_dict'] = new_network_fn_state_dict
                    
                    # save expert in new ckpt
                    output_dict_expert['network_fine_expert_state_dict'] = render_kwargs_train['network_fine'].expert.state_dict()
                    
                # for key in output_dict['network_fine_state_dict'].keys():
                #     print(key)
                # input()

                torch.save(output_dict_expert, path_second)
                print('Saved checkpoints of expert at', path_second)
            
            torch.save(output_dict, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, args=args)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, args=args, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')


    
        if i%args.i_print==0:
            try:
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            except KeyboardInterrupt:
                break
            except:
                pass
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
