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
        # self.args = args

        self.input_linears = nn.ModuleList([nn.Linear(input_dim, W)])
        self.pts_linears = nn.ModuleList([nn.Linear(W, W) for i in range(D)])
        self.output_linears = nn.ModuleList([nn.Linear(W, output_dim)])

        for m in self.input_linears:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                # torch.nn.init.zeros_(m.weight)
                if m.bias is not None:
                    # torch.nn.init.xavier_uniform_(m.bias)
                    torch.nn.init.zeros_(m.bias)

        # for i in range(D):
        for m in self.pts_linears:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                # torch.nn.init.zeros_(m.weight)
                if m.bias is not None:
                    # torch.nn.init.xavier_uniform_(m.bias)
                    torch.nn.init.zeros_(m.bias)
        
        for m in self.output_linears:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                # torch.nn.init.zeros_(m.weight)
                if m.bias is not None:
                    # torch.nn.init.xavier_uniform_(m.bias)
                    torch.nn.init.zeros_(m.bias)
                
    def forward(self, h):
        h = self.input_linears[0](h)

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)

        # outputs = h
        outputs = self.output_linears[0](h)
        
        return outputs

if __name__=='__main__':
    '''
    test adaptor
    '''
    input = torch.randn(size=(32, 256))
    _expert = expert_v2(2, 128, input_dim=256, output_dim=256)
    out = _expert(input)