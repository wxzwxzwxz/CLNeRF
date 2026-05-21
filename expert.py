import torch
import torch.nn as nn
import torch.nn.functional as F

class expert_v2(nn.Module):
    ''' 
    '''
    def __init__(self, D=2, W=256, input_dim=256, output_dim=256, args=None):
        super().__init__()
        self.args = args

        self.input_linears = nn.Linear(input_dim, W)
        self.pts_linears = nn.ModuleList([nn.Linear(W, W) for i in range(D)])
        self.output_linears = nn.Linear(W, output_dim)
        
        if args.use_expert_predict_mask:
            self.mask_linear = nn.Linear(W, 1)

        torch.nn.init.xavier_uniform_(self.input_linears.weight)
        if self.input_linears.bias is not None:
            torch.nn.init.zeros_(self.input_linears.bias)
        
        torch.nn.init.xavier_uniform_(self.output_linears.weight)
        if self.output_linears.bias is not None:
            torch.nn.init.zeros_(self.output_linears.bias)

        for m in self.pts_linears:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                
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
            outputs = F.relu(outputs)
        
        return outputs
    
if __name__=='__main__':
    '''
    test adaptor
    '''
    input = torch.randn(size=(32, 256))
    _expert = expert_v2(2, 128, input_dim=256, output_dim=256)
    out = _expert(input)