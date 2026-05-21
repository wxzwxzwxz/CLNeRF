import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from expert import *

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
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
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
            
        if args.use_expert:
            self.expert = expert_v2(D=args.expert_d, W=args.expert_w, input_dim=W, output_dim=W, args=args)
            
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        self.args = args
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
            if self.args.use_predict_mask and not self.args.use_expert_predict_mask:
                self.mask_linear = nn.Linear(W, 1)
        else:
            if self.args.use_predict_mask and not self.args.use_expert_predict_mask:
                output_ch += 1

            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, return_feat=False, index=None, prev_expert=None):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts

        h = self.pts_linears[0](h)
        h = F.relu(h)

        output_dict = dict()
        if self.args.use_expert:
            expert_h = self.expert(h)

            if self.args.use_teacher_nerf_with_branch and prev_expert is not None:
                expert_h_prev = prev_expert(h)

        for i in range(1, len(self.pts_linears)):
            h = self.pts_linears[i](h)

            if i != len(self.pts_linears)-1:
                h = F.relu(h)

            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.args.use_expert:
            h_teacher = h
            h_teacher = F.relu(h_teacher)

            # merge old_feat and new_feat
            if self.args.use_predict_mask:
                if self.args.use_expert_predict_mask:
                    mask = expert_h[..., -1:]
                    h = torch.sigmoid(mask) * expert_h[..., :-1] + h
                    
                    if self.args.use_teacher_nerf_with_branch and prev_expert is not None:
                        mask_prev = expert_h_prev[..., -1:]
                        h = torch.sigmoid(mask_prev) * expert_h_prev[..., :-1] + h

            # relu or not
            h = F.relu(h)
        else:
            h = F.relu(h)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)

        h = torch.cat([feature, input_views], -1)
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)


        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, alpha], -1)
        if self.args.use_predict_mask:
            outputs = torch.cat([outputs, mask], -1)
        

        if self.args.use_expert:
            with torch.no_grad():
                alpha = self.alpha_linear(h_teacher)
                feature = self.feature_linear(h_teacher)

                h = torch.cat([feature, input_views], -1)
                for i, l in enumerate(self.views_linears):
                    h = self.views_linears[i](h)
                    h = F.relu(h)
                rgb = self.rgb_linear(h)
                output_dict['outputs_teacher'] = torch.cat([rgb, alpha], -1)
            
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

