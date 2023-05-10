import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#for LoRA
import loralib as lora
from adapter import bottle_neck_adapter
from expert import *

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2mse_withmask = lambda x, y, mask : torch.sum((x * mask - y * mask) ** 2) / (torch.sum(mask)+1e-6)
img2bce = nn.BCELoss()
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

img2mse_np = lambda x, y : np.mean((x - y) ** 2)
mse2psnr_np = lambda x : -10. * np.log(x) / np.log(np.array([10.]))

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, args=None):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        if args.lora:
            assert args.lora_layers is not None, 'when using LoRA, lora_layers must be set'
            print("using LoRA")
            rank=args.lora_rank
            if 'pts_linears.0' in args.lora_layers:
                inputs=[lora.Linear(input_ch, W,r=rank)]
            else:
                inputs=[nn.Linear(input_ch, W)]

            middle_layers=[]
            for i in range(1,D):
                if i-1 not in self.skips:
                    _in_ch=W
                else:
                    _in_ch=W + input_ch
                if 'pts_linears.'+str(i) in args.lora_layers: #and 'pts_linears.'+str(i) != 'pts_linears.0':
                    middle_layers.append(lora.Linear(_in_ch, W,r=rank))
                else:
                    middle_layers.append(nn.Linear(_in_ch, W))

            self.pts_linears = nn.ModuleList(
                inputs + middle_layers)
        else:
            self.pts_linears = nn.ModuleList(
                [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
            
        if args.adapter_layers:
            # self.adapters={}
            adapters_list = []
            for i in args.adapter_layers:
                # if int(i)+1 in self.skips:
                #     self.pts_linears.insert(int(i)+1,bottle_neck_adapter(out_dims=W + input_ch))
                # if
                # self.adapters[i] = bottle_neck_adapter()
                adapters_list.append(bottle_neck_adapter(in_dims=256, out_dims=256, bottle_neck_dim=args.bottle_neck_dim))
            
            self.adapters = nn.ModuleList(adapters_list)
            
            #record the adapter layer for skip nonlinearity
            # self.adapter_index=[]
            # for i,layer in enumerate(self.pts_linears):
            #     if 'adapter' in layer._get_name():
            #         self.adapter_index.append(i)
            # print('using adapter',self.adapters)

        if args.use_expert:
            if args.expert_version == 'v1':
                self.expert = expert(D=args.expert_d, W=args.expert_w, args=args)
            elif args.expert_version == 'v2':
                # d + 2
                self.expert = expert_v2(D=args.expert_d, W=args.expert_w, input_dim=W, output_dim=W, args=args)
            elif args.expert_version == 'v3':
                # d + 2
                self.expert = expert_v2(D=args.expert_d, W=args.expert_w, input_dim=input_ch, output_dim=W, args=args)
            elif args.expert_version == 'v4':
                self.expert = expert_v4(D=args.expert_d, W=args.expert_w, input_ch_views=self.input_ch_views, input_dim=input_ch, output_dim=W, args=args)

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        self.args = args
        if self.args.add_dino:
            self.emb_linear_penultimate = nn.Linear(W, W//2)

        if use_viewdirs:
            if args.lora and 'feature_linear' in args.lora_layers:
                self.feature_linear = lora.Linear(W, W, r=rank)
            else:
                self.feature_linear = nn.Linear(W, W)

            if args.lora and 'alpha_linear' in args.lora_layers:
                self.alpha_linear = lora.Linear(W, 1, r=rank)
            else:
                self.alpha_linear = nn.Linear(W, 1)

            if args.lora and 'rgb_linear' in args.lora_layers:
                self.rgb_linear = lora.Linear(W//2, 3, r=rank)
            else:
                self.rgb_linear = nn.Linear(W//2, 3)

            if self.args.add_dino:
                pass
                self.emb_linear = nn.Linear(W//2, 64)
            
            if self.args.use_predict_mask and not self.args.use_expert_predict_mask:
                self.mask_linear = nn.Linear(W, 1)
        else:
            if args.lora and 'output_linear' in args.lora_layers:
                self.output_linear = lora.Linear(W, output_ch, r=rank)
            else:
                if self.args.use_predict_mask and not self.args.use_expert_predict_mask:
                    output_ch += 1

                self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, return_feat=False):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts

        h = self.pts_linears[0](h)
        h = F.relu(h)
        if return_feat == True:
            output_dict = dict()
            output_dict['pts_linears_0'] = h

        if self.args.use_expert:
            if self.args.expert_version == 'v1' or self.args.expert_version == 'v2':
                # expert_h = h
                expert_h = self.expert(h)
            elif self.args.expert_version == 'v3':
                expert_h = self.expert(input_pts)
            elif self.args.expert_version == 'v4':
                expert_h = self.expert(input_pts, input_views)

        for i in range(1, len(self.pts_linears)):
            h = self.pts_linears[i](h)

            if self.args.adapter_layers is not None and str(i) in self.args.adapter_layers:
                if self.args.adapter_version == 0:
                    # new block, residual, best
                    h = self.adapters[i](h)
                elif self.args.adapter_version == 1:
                    # new block v2, residual
                    h = F.relu(h)
                    h = self.adapters[i](h)
                elif self.args.adapter_version == 2:
                    # residual
                    residual = self.adapters[i](h)
                    h = h.add(residual)
                elif self.args.adapter_version == 3:
                    # residual v2
                    h = F.relu(h)
                    residual = self.adapters[i](h)
                    h = h.add(residual)

            if i != len(self.pts_linears)-1:
                h = F.relu(h)

            if return_feat == True:
                output_dict['pts_linears_'+str(i)] = h

            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.args.use_expert:
            # relu or not
            if not self.args.use_expert_predict_mask_merge_relu:
                h = F.relu(h)

            if self.args.use_predict_mask and not self.args.use_expert_predict_mask:
                mask = self.mask_linear(h)

            # merge old_feat and new_feat
            if self.args.expert_version == 'v1' \
                or self.args.expert_version == 'v2' \
                or self.args.expert_version == 'v3':

                if self.args.use_predict_mask:
                    if self.args.use_expert_predict_mask:
                        mask = expert_h[..., -1:]
                        # h = mask * expert_h[..., :-1] + h
                        h = h - mask * expert_h[..., :-1]
                    else:
                        # h = mask * expert_h + (1-mask) * h
                        # h = mask * expert_h + h
                        h = h - mask * expert_h
                else:
                    h += expert_h
            
            # relu or not
            if self.args.use_expert_predict_mask_merge_relu:
                h = F.relu(h)
        else:
            h = F.relu(h)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)

            if return_feat == True:
                output_dict['alpha_linear'] = alpha
                output_dict['feature_linear'] = feature

            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

                if return_feat == True:
                    output_dict['views_linears_'+str(i)] = h

            rgb = self.rgb_linear(h)
            if self.args.add_dino:
                pass
                # h = self.emb_linear_penultimate(feature)
                # ft = self.emb_linear(h)
                outputs = torch.cat([rgb, alpha, ft], -1)
            else:
                outputs = torch.cat([rgb, alpha], -1)

            if self.args.use_expert:
                if self.args.expert_version == 'v4':
                    raise NotImplemented
                    if self.args.use_predict_mask:
                        outputs = mask * expert_h + (1-mask) * outputs
                    else:
                        outputs = outputs + expert_h

            if self.args.use_predict_mask:
                # mask = self.mask_linear(h)
                outputs = torch.cat([outputs, mask], -1)
        else:
            outputs = self.output_linear(h)

        if return_feat == True:
            return outputs, output_dict
        else:
            return outputs # , None   

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def sample_rays(args, hwf, K, i, i_train, N_rand, poses, start, use_batching=False):
    # N_rand = args.N_rand

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Sample random ray batch
    if use_batching:
        pass
    else:
        # Random from one image
        img_i = np.random.choice(i_train)
        # target = images[img_i]
        # target = torch.Tensor(target).to(device)
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
            # target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

    return batch_rays
