import torch
import torch.nn as nn
import torch.nn.functional as F

class expert(nn.Module):
    # def __init__(self,in_dims=256, out_dims=256, bottle_neck_dim=None):
    def __init__(self, D=2, W=256, args=None):
        super().__init__()

        self.D = D
        self.W = W
        # self.input_ch = input_ch
        # self.input_ch_views = input_ch_views
        # self.skips = skips
        # self.use_viewdirs = use_viewdirs
        self.args = args

        # self.pts_linears = nn.ModuleList(
        #         [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        # self.feature_linear = nn.Linear(W, W)
        self.pts_linears = nn.ModuleList([nn.Linear(W, W) for i in range(D)])

        # for i in range(D):
        for m in self.pts_linears:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                # torch.nn.init.zeros_(m.weight)
                if m.bias is not None:
                    # torch.nn.init.xavier_uniform_(m.bias)
                    torch.nn.init.zeros_(m.bias)
                
    # def forward(self, x):
    def forward(self, h):
        # input_pts, _ = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        # h = input_pts
        
        # if return_feat == True:
        #     output_dict = dict()

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            # if return_feat == True:
            #     output_dict['pts_linears_'+str(i)] = h

            # if i in self.skips:
            #     h = torch.cat([input_pts, h], -1)

        # feature = self.feature_linear(h)
        outputs = h
        # print(outputs.mean())
        
        return outputs


class expert_v2(nn.Module):
    ''' 
    '''
    def __init__(self, D=2, W=256, input_dim=256, output_dim=256, args=None):
        super().__init__()
        # self.D = D
        # self.W = W
        self.args = args

        # self.input_linears = nn.ModuleList([nn.Linear(input_dim, W)])
        self.input_linears = nn.Linear(input_dim, W)
        self.pts_linears = nn.ModuleList([nn.Linear(W, W) for i in range(D)])
        # self.output_linears = nn.ModuleList([nn.Linear(W, output_dim)])
        self.output_linears = nn.Linear(W, output_dim)

        if args.use_expert_predict_mask:
            self.mask_linear = nn.Linear(W, 1)

        # torch.nn.init.xavier_uniform_(self.input_linears.weight)
        # if self.input_linears.bias is not None:
        #     torch.nn.init.zeros_(self.input_linears.bias)
        
        # torch.nn.init.xavier_uniform_(self.output_linears.weight)
        # if self.output_linears.bias is not None:
        #     torch.nn.init.zeros_(self.output_linears.bias)

        # for m in self.pts_linears:
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             torch.nn.init.zeros_(m.bias)
                
    def forward(self, h):
        h = self.input_linears(h)

        # newly added
        h = F.relu(h)

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)

        outputs = self.output_linears(h)
        
        if self.args.use_expert_predict_mask:
            mask = self.mask_linear(h)
            if self.args.use_expert_predict_mask_worelu:
                pass
            else:
                outputs = F.relu(outputs)
            outputs = torch.cat([outputs, mask], -1)
        else:
            # newly added
            outputs = F.relu(outputs)
            # print(outputs.mean(), outputs.max(), outputs.min())
            # input()
        
        return outputs
    

class expert_v4(nn.Module):
    ''' Input fts, output color and density
    '''
    def __init__(self, D=2, W=256, input_ch_views=3, input_dim=256, output_dim=256, args=None):
        super().__init__()
        # self.D = D
        # self.W = W
        self.args = args

        # self.input_linears = nn.ModuleList([nn.Linear(input_dim, W)])
        self.input_linears = nn.Linear(input_dim, W)
        self.pts_linears = nn.ModuleList([nn.Linear(W, W) for i in range(D)])
        # self.output_linears = nn.ModuleList([nn.Linear(W, output_dim)])
        
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        self.rgb_linear = nn.Linear(W//2, 3)

        self.output_linears = nn.Linear(W, output_dim)

        # if args.use_expert_predict_mask:
        #     self.mask_linear = nn.Linear(W, 1)

        # torch.nn.init.xavier_uniform_(self.input_linears.weight)
        # if self.input_linears.bias is not None:
        #     torch.nn.init.zeros_(self.input_linears.bias)
        
        # torch.nn.init.xavier_uniform_(self.output_linears.weight)
        # if self.output_linears.bias is not None:
        #     torch.nn.init.zeros_(self.output_linears.bias)

        # for m in self.pts_linears:
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             torch.nn.init.zeros_(m.bias)
                
    def forward(self, h, input_views):
        h = self.input_linears(h)

        # newly added
        h = F.relu(h)

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)

        h = torch.cat([feature, input_views], -1)
        
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, alpha], -1)

        return outputs

if __name__=='__main__':
    '''
    test adaptor
    '''
    input = torch.randn(size=(32, 256))
    _expert = expert_v2(2, 128, input_dim=256, output_dim=256)
    out = _expert(input)