import torch
import torch.nn as nn


class bottle_neck_adapter(nn.Module):
    def __init__(self,in_dims=256,out_dims=256,bottle_neck_dim=None):
        super().__init__()
        '''
        default bottle_neck half the dimension
        '''
        if not bottle_neck_dim:
            bottle_neck_dim=in_dims//2 
        
        self.adapter_down=nn.Linear(in_dims,bottle_neck_dim)
        torch.nn.init.xavier_uniform_(self.adapter_down.weight)

        self.nonlinearity=nn.ReLU()

        self.adapter_up=nn.Linear(bottle_neck_dim,out_dims)
        torch.nn.init.xavier_uniform_(self.adapter_up.weight)
    

    def forward(self,x):
        '''
        h=W_up*f(W_down*h)+residual
        '''
        _x=x.clone()
        x= self.adapter_up(self.nonlinearity(self.adapter_down(x))) + x
        return x
    



if __name__=='__main__':
    '''
    test adaptor
    '''
    input = torch.randn(size=(32,256))
    _adapter=bottle_neck_adapter(256,256,32)
    out=_adapter(input)

    pass
