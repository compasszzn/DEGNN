import torch
from torch import nn
from models.models.gcl import E_GCL_vel
import numpy as np

import torch
import torch.nn as nn
import numpy as np



class DEGNN_vel(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf,model_type,pool_method,device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0,
                 recurrent=False, norm_diff=False, tanh=False,embed_vel=True):
        super(DEGNN_vel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.module = E_GCL_vel(self.hidden_nf, self.hidden_nf, self.hidden_nf,model_type,pool_method,edges_in_d=in_edge_nf, device=device, 
                                                    act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent,
                                                    norm_diff=norm_diff, tanh=tanh ,embed_vel=embed_vel)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_vel(self.hidden_nf, self.hidden_nf, self.hidden_nf,model_type,pool_method, edges_in_d=in_edge_nf,
                                                    act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent,
                                                    norm_diff=norm_diff, tanh=tanh ,embed_vel=embed_vel))

        self.to(self.device)

    def forward(self, h, x, edges, vel, edge_attr):
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, vel, edge_attr=edge_attr)
        #     print(x)
        # print('---------------------------')
        return x
