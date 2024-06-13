from torch import nn
import torch
import torch.nn.functional as F


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


# add our model here
class GMNLayer(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, norm_diff=False, tanh=False,
                 learnable=False):
        """
        The layer of Graph Mechanics Networks.
        :param input_nf: input node feature dimension
        :param output_nf: output node feature dimension
        :param hidden_nf: hidden dimension
        :param edges_in_d: input edge dimension
        :param nodes_att_dim: attentional dimension, inherited
        :param act_fn: activation function
        :param recurrent: residual connection on x
        :param coords_weight: coords weight, inherited
        :param attention: use attention on edges, inherited
        :param norm_diff: normalize the distance, inherited
        :param tanh: Tanh activation, inherited
        :param learnable: use learnable FK
        """
        super(GMNLayer, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.learnable = learnable
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        n_basis_stick = 1
        n_basis_hinge = 3

        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))
        self.coord_mlp_w_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))
        self.center_mlp = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, input_nf))

        self.f_stick_mlp = nn.Sequential(
            nn.Linear(n_basis_stick * n_basis_stick, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, n_basis_stick)
        )
        self.f_hinge_mlp = nn.Sequential(
            nn.Linear(n_basis_hinge * n_basis_hinge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, n_basis_hinge)
        )

        if self.learnable:
            self.stick_v_fk_mlp = nn.Sequential(
                nn.Linear(3 * 3, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, 3)
            )
            self.hinge_v_fk_mlp = nn.Sequential(
                nn.Linear(3 * 3, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, 3)
            )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

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

    def node_model(self, x, edge_index, edge_attr, node_attr, others=None):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        if others is not None:  # can concat h here
            agg = torch.cat([others, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        f = agg * self.coords_weight
        return coord + f, f

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def update(self, x, v, f, h):
        """
        Update X and V given the current X, V, and force F
        :param x: position  [N, 3]
        :param v: velocity  [N, 3]
        :param f: force  [N, 3]
        :param h:  node feature  [N, n_hidden]
        :param node_index: [K, 2] for stick, [K, 3] for hinge, [K,] for isolated, K is the number of rigid objects
        :param type:  the type of rigid objects, 'Isolated' or 'Stick' or 'Hinge'
        :return: the updated x [N, 3] and v [N, 3]
        """
        _x, _v, _f, _h = x, v, f, h
        _a = _f / 1.
        _v = self.coord_mlp_vel(_h) * _v + _a
        _x = _x + _v
        # put the updated x, v (local object) back to x, v (global graph)
        x = _x
        v = _v
        return x, v


    def forward(self, h, edge_index, x, v, edge_attr=None, node_attr=None):
        """
        :param h: the node aggregated feature  [N, n_hidden]
        :param edge_index:  [2, M], M is the number of edges
        :param x: input coordinate  [N, 3]
        :param v: input velocity  [N, 3]
        :param cfg: {'isolated': idx, 'stick': [(c0, c1) ...] (K, 2), 'hinge': [(c0, c1, c2) ...] (K, 3)}. K is the number of rigid obj
        :param edge_attr: edge feature  [M, n_edge]
        :param node_attr: the node input feature  [N, n_in]
        :return: the updated h, x, v, and edge_attr
        """

        # aggregate force (equivariant message passing on the whole graph)
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, x)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)  # [B*M, Ef], the global invariant message
        _, f = self.coord_model(x, edge_index, coord_diff, edge_feat)  # [B*N, 3]
        x, v = self.update(x, v, f, h)

        # for type in cfg:
        #     x, v = self.update(x, v, f, h, node_index=cfg[type], type=type)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr, others=h)

        return h, x, v, edge_attr

    @staticmethod
    def compute_rotation_matrix(theta, d):
        x, y, z = torch.unbind(d, dim=-1)
        cos, sin = torch.cos(theta), torch.sin(theta)
        ret = torch.stack((
            cos + (1 - cos) * x * x,
            (1 - cos) * x * y - sin * z,
            (1 - cos) * x * z + sin * y,
            (1 - cos) * x * y + sin * z,
            cos + (1 - cos) * y * y,
            (1 - cos) * y * z - sin * x,
            (1 - cos) * x * z - sin * y,
            (1 - cos) * y * z + sin * x,
            cos + (1 - cos) * z * z,
        ), dim=-1)

        return ret.reshape(-1, 3, 3)  # [B*N, 3, 3]

class GNSLayer(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_nf=1, act_fn=nn.ReLU(), bias=True, recurrent=True):
        super(GNSLayer, self).__init__()

        self.recurrent = recurrent
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 3, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf))

        self.gate = nn.Sequential(
            nn.Linear(hidden_nf * 3, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
            nn.Sigmoid())
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 2, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

    def edge_model(self, source, target, e):
        input = torch.cat([source, target, e], dim=1)
        gate = self.gate(input)
        edge_embedding = self.edge_mlp(input) + e
        return edge_embedding       

    def node_model(self, h, edge_index, e):
        row, col = edge_index
        agg = unsorted_segment_sum(e, row, num_segments=h.size(0))
        out = torch.cat([h, agg], dim=1)
        out = self.node_mlp(out)
        if self.recurrent:
            out = out + h
        return out

    def forward(self, h, e, edge_index, edge_attr=None):
        row, col = edge_index
        edge_feat = self.edge_model(h[row], h[col], e)
        h = self.node_model(h, edge_index, e)
        return h, edge_feat
class CrowdSimLayer(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_nf=1, act_fn=nn.ReLU(), bias=True, recurrent=True):
        super(CrowdSimLayer, self).__init__()

        self.recurrent = recurrent
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 3, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf))

        self.gate = nn.Sequential(
            nn.Linear(hidden_nf * 3, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
            nn.Sigmoid())
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 2, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

    def edge_model(self, source, target, e):
        input = torch.cat([source, target, e], dim=1)
        gate = self.gate(input)
        edge_embedding = self.edge_mlp(input) * gate + e
        return edge_embedding         

    def node_model(self, h, edge_index, e):
        row, col = edge_index
        agg = unsorted_segment_sum(e, row, num_segments=h.size(0))
        out = torch.cat([h, agg], dim=1)
        out = self.node_mlp(out)
        if self.recurrent:
            out = out + h
        return out

    def forward(self, h, e, edge_index, edge_attr=None):
        row, col = edge_index
        edge_feat = self.edge_model(h[row], h[col], e)
        h = self.node_model(h, edge_index, e)
        return h, edge_feat