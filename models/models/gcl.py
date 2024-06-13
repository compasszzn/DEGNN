from torch import nn, autograd
import torch
import torch.nn.functional as F
import itertools
from scipy.spatial.transform import Rotation as R
import numpy as np
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import softmax, dense_to_sparse, add_remaining_self_loops
class E_GCL(nn.Module):####
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False,embed_vel=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1
        vel_nf = 6
        self.embed_vel=embed_vel

        if self.embed_vel:
            self.edge_mlp = nn.Sequential(
                nn.Linear(input_edge + edge_coords_nf + edges_in_d + vel_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn)
        else:
            self.edge_mlp = nn.Sequential(
                nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 3, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        #if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)



    def edge_model(self, source, target, radial, vel_rot,edge_attr):

        if self.embed_vel:
            if vel_rot is None:
                if edge_attr is None:  # Unused.
                    out = torch.cat([source, target, radial], dim=1)
                else:
                    out = torch.cat([source, target, radial,edge_attr], dim=1)
            else:    
                if edge_attr is None:  # Unused.
                    out = torch.cat([source, target, radial,vel_rot], dim=1)
                else:
                    out = torch.cat([source, target, radial,vel_rot,edge_attr], dim=1)
        else:            
            if edge_attr is None:  # Unused.
                out = torch.cat([source, target, radial], dim=1)
            else:
                out = torch.cat([source, target, radial,edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg
    
    

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord = coord + agg*self.coords_weight
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        radial = torch.cat([coord[row], coord[col], radial], dim=1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial,vel_rot=None, edge_attr=edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr

class DE_GCL_2D(nn.Module):####
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(DE_GCL_2D, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1
        vel_nf = 6


        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d + vel_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 2, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        #if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)

    def edge_model(self, source, target, radial, vel_rot,edge_attr):
        if vel_rot is None:
            if edge_attr is None:  # Unused.
                out = torch.cat([source, target, radial], dim=1)
            else:
                out = torch.cat([source, target, radial,edge_attr], dim=1)
        else:
            if edge_attr is None:  # Unused.
                out = torch.cat([source, target, radial,vel_rot], dim=1)
            else:
                out = torch.cat([source, target, radial,vel_rot,edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg
    

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord = coord + agg*self.coords_weight
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        radial = torch.cat([coord[row], coord[col], radial], dim=1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr

class E_GCL_vel(E_GCL):###
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf,model_type,pool_method, edges_in_d=0, device='cpu', nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, norm_diff=False, tanh=False,embed_vel=True):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_att_dim, act_fn=act_fn,
                       recurrent=recurrent, coords_weight=coords_weight, attention=attention, norm_diff=norm_diff, tanh=tanh,embed_vel=embed_vel)
        self.norm_diff = norm_diff
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn, 
            nn.Linear(hidden_nf, 1))
        self.model_type=model_type

        rotate_x_90 = torch.FloatTensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        rotate_y_90 = torch.FloatTensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        rotate_z_90 = torch.FloatTensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        rotate_x_180 = torch.FloatTensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rotate_y_180 = torch.FloatTensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        rotate_z_180 = torch.FloatTensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

        reflect_x = torch.FloatTensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        reflect_y = torch.FloatTensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        reflect_z = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        reflect_c = torch.FloatTensor([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])

        if self.model_type in ['nbody_charged_5_5_5','nbody_charged_4_4_4','nbody_gravity_5_5_5','nbody_gravity_4_4_4']:
            basic_ops = [rotate_x_90, rotate_y_90, rotate_z_90, reflect_x, reflect_y, reflect_z]
        elif self.model_type in ['nbody_charged_5_4_4','nbody_gravity_5_4_4']:
            basic_ops = [rotate_x_90,rotate_y_180,rotate_z_180, reflect_x, reflect_y, reflect_z]
        elif self.model_type in ['nbody_charged_5_4_3','nbody_gravity_5_4_3']:
            basic_ops = [rotate_x_180,rotate_y_180,rotate_z_180,reflect_x, reflect_y, reflect_z]
        elif self.model_type in ['lipo']:
            basic_ops = [rotate_x_180,rotate_y_180,rotate_z_180,reflect_x, reflect_y, reflect_z]
        elif self.model_type in ['lips']:
            basic_ops = [reflect_c]
        else:
            raise Exception("Unknown dataset")



        D4h_group = [torch.eye(3)]  
        for _ in range(5): 
            new_elements = []
            for op in basic_ops:
                for element in D4h_group:
                    new_element = torch.mm(op, element)
                    new_elements.append(new_element)
            D4h_group.extend(new_elements)     
        unique_ops = []
        for op in D4h_group:
            if not any(torch.all(torch.eq(op, unique_op)) for unique_op in unique_ops):
                unique_ops.append(op)

        print(len(unique_ops))
        self.group = [op.to(device) for op in unique_ops]
        
        self.pool_method=pool_method

        if self.pool_method == 'self_attn':
            self.attn=nn.MultiheadAttention(hidden_nf,1)
        elif self.pool_method == 'spa':
            self.attn=nn.MultiheadAttention(hidden_nf,1)
        elif self.pool_method == 'space_attn':
            self.attn=nn.MultiheadAttention(hidden_nf,1)
        elif self.pool_method == 'space':
            self.attn=nn.MultiheadAttention(hidden_nf,1)
            # self.lamb=nn.Linear(2*hidden_nf+1,1)
            self.lamb = nn.Sequential(
            nn.Linear(2*hidden_nf+1, 1),
            act_fn)
        elif self.pool_method == 'space_res_attn':
            self.attn=nn.MultiheadAttention(hidden_nf+7,1)
        elif self.pool_method == 'mlp':
            self.attn=nn.Linear(hidden_nf,1)
        elif self.pool_method == 'mlp_soft':
            self.attn=nn.Linear(hidden_nf,1)
        elif self.pool_method == 'mlp_attn':
            self.lin=nn.Linear(hidden_nf,1)
            self.attn=nn.MultiheadAttention(hidden_nf,1)
        elif self.pool_method == 'read_out_mean_max':
            self.pool_model=nn.Sequential(
            act_fn,
            nn.Linear(2*hidden_nf, hidden_nf))
        elif self.pool_method == 'sort':
            self.attn=nn.Linear(len(self.group)*(3*4+1),3*4+1)

    def forward(self, h, edge_index, coord, vel, edge_attr=None, node_attr=None):
        row, col = edge_index


        _, coord_diff = self.coord2radial(edge_index, coord)

        if self.pool_method=='mean':
            edge_feat = 0
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                edge_feat += self.edge_model(h[row], h[col], radial,vel_rot, edge_attr)
            edge_feat /= len(self.group)

        elif self.pool_method=='sort':

            cat_edge_features = []
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                cat_edge_features.append(torch.cat([radial,vel_rot], dim=1))
            catedge = torch.stack(cat_edge_features).permute(1,0,2)
            sum_catedge = torch.sum(catedge, dim=2)
            sorted_indices = torch.argsort(sum_catedge)
            # sorted_indices = torch.argsort(catedge[:, :, 0], dim=1)
            catedge = torch.gather(catedge, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, catedge.size(-1)))
            num_slices = len(self.group)
            catedge = torch.cat([catedge[:, i, :] for i in range(num_slices)], dim=1)
            catedge=self.attn(catedge)
            edge_feat=self.edge_model(h[row], h[col], catedge,vel_rot=None,edge_attr= edge_attr)

        elif self.pool_method=='rank':
            rank =1
            cat_edge_features = []
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                cat_edge_features.append(torch.cat([radial,vel_rot], dim=1))

            catedge = torch.stack(cat_edge_features).permute(1,0,2)
            sorted_indices = torch.argsort(catedge[:, :, 0], dim=1)
            sorted_data = torch.gather(catedge, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, catedge.size(-1)))
            sorted_cat_edge_features=sorted_data[:,rank,:]
            edge_feat=self.edge_model(h[row], h[col], sorted_cat_edge_features,vel_rot=None,edge_attr= edge_attr)


        elif self.pool_method=='sum':
            edge_feat = 0
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                edge_feat += self.edge_model(h[row], h[col], radial,vel_rot, edge_attr)


        elif self.pool_method=='self_attn':
            list_edge_features = []
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                list_edge_features.append(self.edge_model(h[row], h[col], radial,vel_rot, edge_attr))
            feat=torch.stack(list_edge_features)
            attn_output, attn_output_weights = self.attn(feat, feat, feat)
            edge_feat=torch.mean(attn_output,dim=0)
        elif self.pool_method=='space_attn':
            boundary = torch.tensor([5,5,5]).to(h.device)
            coord_source = boundary - coord[row]
            coord_target = boundary - coord[col]
            list_edge_features = []
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                list_edge_features.append(self.edge_model(h[row], h[col], radial,vel_rot, edge_attr,coord_source,coord_target))
            feat=torch.stack(list_edge_features)
            attn_output, attn_output_weights = self.attn(feat, feat, feat)
            edge_feat=torch.mean(attn_output,dim=0)
        elif self.pool_method=='space':
            list_edge_features = []
            for e in self.group:
                radial, coord_diff = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                list_edge_features.append(self.edge_model(h[row], h[col], radial,vel_rot, edge_attr))
            feat=torch.stack(list_edge_features)
            attn_output, attn_output_weights = self.attn(feat, feat, feat)
            edge_feat=torch.mean(attn_output,dim=0)
            lam=self.lamb(torch.cat([h[row], h[col],torch.sum((coord_diff)**2, 1).unsqueeze(1)],dim=1))
            edge_feat=lam*edge_feat

        elif self.pool_method=='mlp':
            list_edge_features = []
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                list_edge_features.append(self.edge_model(h[row], h[col], radial,vel_rot, edge_attr))
            feat=torch.stack(list_edge_features)
            edge_feat=torch.mean(self.attn(feat)*feat,dim=0)
        elif self.pool_method=='mlp_soft':
            list_edge_features = []
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                list_edge_features.append(self.edge_model(h[row], h[col], radial,vel_rot, edge_attr))
            feat=torch.stack(list_edge_features)
            edge_feat=torch.sum(F.softmax(self.attn(feat))*feat,dim=0)
        elif self.pool_method=='mlp_attn':
            list_edge_features = []
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                list_edge_features.append(self.edge_model(h[row], h[col], radial,vel_rot, edge_attr))
            feat=torch.stack(list_edge_features)
            feat=F.softmax(self.lin(feat))*feat
            attn_output, attn_output_weights = self.attn(feat, feat, feat)
            edge_feat=torch.mean(attn_output,dim=0)

        elif self.pool_method=='read_out_mean_max':
            list_edge_features = []
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                list_edge_features.append(self.edge_model(h[row], h[col], radial,vel_rot, edge_attr))
            mean_edge_feat = 0
            for feat in list_edge_features:
                mean_edge_feat+=feat
            mean_edge_feat /= len(self.group)
            max_edge_feat = torch.squeeze(torch.max(torch.stack(list_edge_features), dim=0, keepdim=True).values)
            mean_max=torch.concatenate([mean_edge_feat,max_edge_feat],dim=1)
            edge_feat = self.pool_model(mean_max)


        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        coord = coord + self.coord_mlp_vel(h) * vel


        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr
    
class DE_GCL_vel_2D(DE_GCL_2D):####
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf,model_type,pool_method,edges_in_d=0, device='cpu', nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, norm_diff=False, tanh=False):
        DE_GCL_2D.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_att_dim, act_fn=act_fn,
                       recurrent=recurrent, coords_weight=coords_weight, attention=attention, norm_diff=norm_diff, tanh=tanh)
        self.norm_diff = norm_diff
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn, 
            nn.Linear(hidden_nf, 1))
        self.model_type=model_type
        rotate_90 = torch.FloatTensor([[0, 1], [-1, 0]])
        rotate_180 = torch.FloatTensor([[-1, 0], [0, -1]])
        reflect_x = torch.FloatTensor([[-1, 0], [0, 1]])
        reflect_y = torch.FloatTensor([[1, 0], [0, -1]])

        basic_ops = [rotate_180,reflect_x,reflect_y]


        D4h_group = [torch.eye(2)]  
        for _ in range(5): 
            new_elements = []
            for op in basic_ops:
                for element in D4h_group:
                    new_element = torch.mm(op, element)
                    new_elements.append(new_element)
            D4h_group.extend(new_elements)     
        unique_ops = []
        for op in D4h_group:
            if not any(torch.all(torch.eq(op, unique_op)) for unique_op in unique_ops):
                unique_ops.append(op)

        print(len(unique_ops))
        self.group = [op.to(device) for op in unique_ops]
        
        self.pool_method=pool_method


        if self.pool_method == 'read_out_mean_max':
            self.pool_model=nn.Sequential(
            act_fn,
            nn.Linear(2*hidden_nf, hidden_nf))
        elif self.pool_method == 'graph_pooling':
            self.calc_information_score = NodeInformationScore()
        elif self.pool_method == 'self_attn':
            self.attn=nn.MultiheadAttention(hidden_nf,1)
        elif self.pool_method == 'space_attn':
            self.attn=nn.MultiheadAttention(hidden_nf,1)
        elif self.pool_method == 'space_res_attn':
            self.attn=nn.MultiheadAttention(hidden_nf+5,1)
        elif self.pool_method == 'mlp':
            self.attn=nn.Linear(hidden_nf,1)
        elif self.pool_method == 'space':
            self.attn=nn.MultiheadAttention(hidden_nf+7,1)
        elif self.pool_method == 'sort':
            self.attn=nn.Linear(len(self.group)*(2*4+1),2*4+1)




    def forward(self, h, edge_index, coord, vel, edge_attr=None, node_attr=None):
        row, col = edge_index


        _, coord_diff = self.coord2radial(edge_index, coord)

        if self.pool_method=='mean':
            edge_feat = 0
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                edge_feat += self.edge_model(h[row], h[col], radial,vel_rot, edge_attr)
            edge_feat /= len(self.group)

        elif self.pool_method=='rank':
            rank =1
            cat_edge_features = []
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                cat_edge_features.append(torch.cat([radial,vel_rot], dim=1))
            #rank

            catedge = torch.stack(cat_edge_features).permute(1,0,2)
            sorted_indices = torch.argsort(catedge[:, :, 0], dim=1)
            sorted_data = torch.gather(catedge, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, catedge.size(-1)))
            sorted_cat_edge_features=sorted_data[:,rank,:]
            edge_feat=self.edge_model(h[row], h[col], sorted_cat_edge_features,vel_rot=None,edge_attr= edge_attr)

        elif self.pool_method=='sort':

            cat_edge_features = []
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                cat_edge_features.append(torch.cat([radial,vel_rot], dim=1))

            catedge = torch.stack(cat_edge_features).permute(1,0,2)
            sum_catedge = torch.sum(catedge, dim=2)
            sorted_indices = torch.argsort(sum_catedge)
            # sorted_indices = torch.argsort(catedge[:, :, 0], dim=1)
            catedge = torch.gather(catedge, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, catedge.size(-1)))
            num_slices = len(self.group)
            catedge = torch.cat([catedge[:, i, :] for i in range(num_slices)], dim=1)
            catedge=self.attn(catedge)
            edge_feat=self.edge_model(h[row], h[col], catedge,vel_rot=None,edge_attr= edge_attr)

        elif self.pool_method=='sum':
            edge_feat = 0
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                edge_feat += self.edge_model(h[row], h[col], radial,vel_rot, edge_attr)
                
        elif self.pool_method=='read_out_mean_max':
            list_edge_features = []
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                list_edge_features.append(self.edge_model(h[row], h[col], radial,vel_rot, edge_attr))
            mean_edge_feat = 0
            for feat in list_edge_features:
                mean_edge_feat+=feat
            mean_edge_feat /= len(self.group)
            max_edge_feat = torch.squeeze(torch.max(torch.stack(list_edge_features), dim=0, keepdim=True).values)
            mean_max=torch.concatenate([mean_edge_feat,max_edge_feat],dim=1)
            edge_feat = self.pool_model(mean_max)
        elif self.pool_method=='graph_pooling':
            list_edge_features = []
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                list_edge_features.append(self.edge_model(h[row], h[col], radial,vel_rot, edge_attr))
        elif self.pool_method=='self_attn':
            list_edge_features = []
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                list_edge_features.append(self.edge_model(h[row], h[col], radial,vel_rot, edge_attr))
            feat=torch.stack(list_edge_features)
            attn_output, attn_output_weights = self.attn(feat, feat, feat)
            edge_feat=torch.mean(attn_output,dim=0)
        elif self.pool_method=='space_attn':
            list_edge_features_v = []
            list_edge_features_q_k = []
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                edge_embed=self.edge_model(h[row], h[col], radial,vel_rot, edge_attr)
                list_edge_features_v.append(edge_embed)
                list_edge_features_q_k.append(torch.cat([edge_embed,radial],dim=1))
            feat_v=torch.stack(list_edge_features_v)
            feat_q_k=torch.stack(list_edge_features_q_k)
            attn_output, attn_output_weights = self.attn(feat_q_k,feat_q_k,feat_q_k)
            group_weight=torch.mean(attn_output_weights,dim=1)
            edge_feat=torch.bmm(feat_v.permute(1,2,0),group_weight.unsqueeze(-1)).squeeze(-1)
        elif self.pool_method=='space_res_attn':
            list_edge_features_v = []
            list_edge_features_q_k = []
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                edge_embed=self.edge_model(h[row], h[col], radial,vel_rot, edge_attr)
                list_edge_features_v.append(edge_embed)
                list_edge_features_q_k.append(torch.cat([edge_embed,radial],dim=1))
            feat_v=torch.stack(list_edge_features_v)
            feat_q_k=torch.stack(list_edge_features_q_k)
            attn_output, attn_output_weights = self.attn(feat_q_k,feat_q_k,feat_q_k)
            group_weight=torch.mean(attn_output_weights,dim=1)
            edge_feat=(torch.bmm(feat_v.permute(1,2,0),group_weight.unsqueeze(-1)).squeeze(-1)+torch.mean(feat_v,dim=0))/2
        elif self.pool_method=='mlp':
            list_edge_features = []
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                list_edge_features.append(self.edge_model(h[row], h[col], radial,vel_rot, edge_attr))
            feat=torch.stack(list_edge_features)
            group_weight=F.softmax(self.attn(feat.permute(1,0,2)),1)
            edge_feat=(torch.bmm(feat.permute(1,2,0),group_weight).squeeze(-1)+torch.mean(feat,dim=0))/2
        elif self.pool_method=='space':
            list_edge_features_v = []
            list_edge_features_q_k = []
            for e in self.group:
                radial, _ = self.coord2radial(edge_index, torch.matmul(coord, e.to(h.device)))
                vel_rot=torch.cat([torch.matmul(vel,e.to(h.device))[row], torch.matmul(vel,e.to(h.device))[col]], dim=1)
                edge_embed=self.edge_model(h[row], h[col], radial,vel_rot, edge_attr)
                list_edge_features_v.append(edge_embed)
                list_edge_features_q_k.append(torch.cat([edge_embed,radial],dim=1))
            feat_v=torch.stack(list_edge_features_v)
            feat_q_k=torch.stack(list_edge_features_q_k)
            _, attn_output_weights = self.attn(feat_q_k,feat_q_k,feat_q_k)
            attn_output=torch.bmm(attn_output_weights,feat_v.permute(1,0,2)).permute(1,0,2)
            edge_feat=torch.mean(attn_output,dim=0)





   

######
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        coord = coord + self.coord_mlp_vel(h) * vel
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr

def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1)).to(data.device)
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1)).to(data.device)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class E_GCL_ERGN(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False,
                 norm_vel=True):
        super(E_GCL_ERGN, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1

        self.norm_vel = norm_vel

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        #if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        # if self.recurrent:
        #     out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        return agg*self.coords_weight

    def coord2radial(self, edge_index, coord, vel):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        vel_diff = vel[row] - vel[col]
        f_x = torch.sum((coord_diff)**2, 1).sqrt().unsqueeze(1)
        f_v = torch.sum((vel_diff)**2, 1).sqrt().unsqueeze(1)

        coord_norm = torch.norm(coord_diff, 2, dim=-1).unsqueeze(1)
        if self.norm_vel:
            vel_norm = torch.norm(vel_diff, 2, dim=-1).unsqueeze(1)
            f_xv = torch.sum((coord_diff/coord_norm) * (vel_diff/vel_norm), 1).unsqueeze(1)
        else:
            f_xv = torch.sum((coord_diff/coord_norm) * vel_diff, 1).unsqueeze(1)

        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        # radial = torch.cat((f_x, f_v, f_xv), dim = 1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr


class E_GCL_ERGN_vel(E_GCL_ERGN):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, norm_diff=False, tanh=False, norm_vel=True):
        E_GCL_ERGN.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_att_dim, act_fn=act_fn,
                       recurrent=recurrent, coords_weight=coords_weight, attention=attention, norm_diff=norm_diff, tanh=tanh,
                       norm_vel=norm_vel)
        self.norm_diff = norm_diff
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))

    def forward(self, h, edge_index, coord, vel, vel_init, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord, vel)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        agg = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        v = self.coord_mlp_vel(h) * vel + agg
        coord = coord + v
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, v, edge_attr