import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
from utils import load_features
import random

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(args, basedir, half_res=False, testskip=1, 
                    load_imgs=True, ori_H=None, ori_W=None, ext='.png',
                    transforms_train=None, transforms_val=None, transforms_test=None, trainskip=1, spherical_radius=4.0,
                    transforms_train_ratio=None, transforms_test_ratio=None):
    splits = ['train', 'val', 'test']
    metas = {}

    for s in splits:
        if s == 'train' and transforms_train is not None and isinstance(transforms_train, list):
            output_dict = None
            for idx in range(len(transforms_train)):
                # print(transforms_train[idx])
                with open(os.path.join(basedir, transforms_train[idx]), 'r') as fp:
                    cur_dict = json.load(fp)
                    if transforms_train_ratio is not None and int(transforms_train_ratio[idx]) > 0:
                        cur_dict["frames"] = random.sample(cur_dict["frames"], int(transforms_train_ratio[idx]))
                    
                    if output_dict is None:
                        output_dict = cur_dict
                    else:
                        output_dict["frames"] += cur_dict["frames"]

                # print(output_dict)
                # input()
            # print(output_dict)
            # input()
            metas[s] = output_dict
            continue
        elif s == 'test' and transforms_test is not None and args.transforms_test_key is not None:
            output_dict = dict()
            with open(os.path.join(basedir, transforms_test[idx]), 'r') as fp:
                cur_dict = json.load(fp)
                for key in cur_dict:
                    if "frames" not in key:
                        output_dict[key] = cur_dict[key]
                output_dict["frames"] = list()
                for frame_dict in cur_dict["frames"]:
                    if args.transforms_test_key in frame_dict["file_path"]:
                        output_dict["frames"].append(frame_dict)

                if transforms_test_ratio is not None:
                    output_dict["frames"] = random.sample(output_dict["frames"], int(transforms_test_ratio))
            
            metas[s] = output_dict
            continue
        elif s == 'test' and transforms_test is not None and isinstance(transforms_test, list):
            output_dict = None
            for idx in range(len(transforms_test)):
                with open(os.path.join(basedir, transforms_test[idx]), 'r') as fp:
                    cur_dict = json.load(fp)
                    if transforms_test_ratio is not None and int(transforms_test_ratio[idx]) > 0:
                        cur_dict["frames"] = random.sample(cur_dict["frames"], int(transforms_test_ratio[idx]))
                    
                    if output_dict is None:
                        output_dict = cur_dict
                    else:
                        output_dict["frames"] += cur_dict["frames"]
            metas[s] = output_dict
            continue
        elif s == 'train' and transforms_train is not None: 
            json_file = transforms_train
        elif s == 'val' and transforms_val is not None: 
            json_file = transforms_val
        elif s == 'test' and transforms_test is not None: 
            json_file = transforms_test
        else:
            json_file = 'transforms_{}.json'.format(s)
        
        with open(os.path.join(basedir, json_file), 'r') as fp:
            metas[s] = json.load(fp)

    if load_imgs == True:
        all_imgs = []
        all_poses = []
        all_paths = []
        counts = [0]
        for s in splits:
            meta = metas[s]
            imgs = []
            poses = []
            paths = []
            
            if s=='train' or trainskip!=1:
                skip = trainskip
            elif s=='train' or testskip==0:
                skip = 1
            else:
                skip = testskip
                
            for frame in meta['frames'][::skip]:
                fname = os.path.join(basedir, frame['file_path'] + ext)
                # print(fname)
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame['transform_matrix']))
                # paths.append(frame['file_path'] + ext)
                all_paths.append(frame['file_path'] + ext)

            imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
            poses = np.array(poses).astype(np.float32)
            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_poses.append(poses)
            
            # all_paths.append(paths)
        
        i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
        
        imgs = np.concatenate(all_imgs, 0)
        poses = np.concatenate(all_poses, 0)
        all_paths = np.array(all_paths)
        
        ori_H, ori_W = imgs[0].shape[:2]
        H = ori_H
        W = ori_W
        # H, W = imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        
        render_poses = torch.stack([pose_spherical(angle, -30.0, spherical_radius) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        
        if half_res:
            H = H//2
            W = W//2
            focal = focal/2.
            
            imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
            imgs_half_res[:, :, :, 3] = 1
            
            for i, img in enumerate(imgs):
                if img.shape[2] == 3:
                    imgs_half_res[i][:, :, :3] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                else:
                    imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

            imgs = imgs_half_res

        if args.scene_scale is not None:
            poses[:, :, :3] *= args.scene_scale
            render_poses[:, :, :3] *= args.scene_scale

        if args.add_dino:
            # dsid = args.dsid
            # fts = load_features(vid=dsid, imhw=(height, width))
            fts = load_features(os.path.join(args.datadir, 'train', 'dino.pt'), imhw=(H, W))
            fns = [x.split('/')[-1] for x in all_paths[i_split[0]]]
            fts_train = np.stack([fts[fn].permute(1,2,0).numpy() for fn in fns], axis=-1)

            # fts = load_features(os.path.join(args.datadir, 'val', 'dino.pt'), imhw=(H, W))
            # fns = [x.split('/')[-1] for x in all_paths[i_split[1]]]
            # fts_val = np.stack([fts[fn].permute(1,2,0).numpy() for fn in fns], axis=-1)

            fts = load_features(os.path.join(args.datadir, 'test', 'dino.pt'), imhw=(H, W))
            fns = [x.split('/')[-1] for x in all_paths[i_split[2]]]
            fts_test = np.stack([fts[fn].permute(1,2,0).numpy() for fn in fns], axis=-1)
        else:
            fts_train = None
            fts_val = None
            fts_test = None

        return imgs, poses, render_poses, [H, W, focal], i_split, all_paths, ori_H, ori_W, fts_train, fts_test
    else:
        all_poses = []
        all_paths = []
        counts = [0]
        for s in splits:
            meta = metas[s]
            poses = []
            
            if s=='train' or trainskip!=1:
                skip = trainskip
            elif s=='train' or testskip==0:
                skip = 1
            else:
                skip = testskip
                
            for frame in meta['frames'][::skip]:
                fname = os.path.join(basedir, frame['file_path'] + ext)
                poses.append(np.array(frame['transform_matrix']))
                all_paths.append(frame['file_path'] + ext)

            poses = np.array(poses).astype(np.float32)
            # counts.append(counts[-1] + imgs.shape[0])
            counts.append(counts[-1] + poses.shape[0])
            all_poses.append(poses)
        
        i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
        
        poses = np.concatenate(all_poses, 0)
        all_paths = np.array(all_paths)
        
        H, W = ori_H, ori_W # int(meta['img_h']), int(meta['img_w'])
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        
        render_poses = torch.stack([pose_spherical(angle, -30.0, spherical_radius) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        
        if half_res:
            H = H//2
            W = W//2
            focal = focal/2.

        if args.add_dino:
            # dsid = args.dsid
            # fts = load_features(vid=dsid, imhw=(height, width))
            fts = load_features(os.path.join(args.datadir, 'train', 'dino.pt'), imhw=(H, W))
            fns = [x.split('/')[-1] for x in all_paths[i_split[0]]]
            fts_train = np.stack([fts[fn].permute(1,2,0).numpy() for fn in fns], axis=-1)

            # fts = load_features(os.path.join(args.datadir, 'val', 'dino.pt'), imhw=(H, W))
            # fns = [x.split('/')[-1] for x in all_paths[i_split[1]]]
            # fts_val = np.stack([fts[fn].permute(1,2,0).numpy() for fn in fns], axis=-1)

            fts = load_features(os.path.join(args.datadir, 'test', 'dino.pt'), imhw=(H, W))
            fns = [x.split('/')[-1] for x in all_paths[i_split[2]]]
            fts_test = np.stack([fts[fn].permute(1,2,0).numpy() for fn in fns], axis=-1)
        else:
            fts_train = None
            fts_val = None
            fts_test = None

        return poses, render_poses, [H, W, focal], i_split, all_paths, fts_train, fts_test


