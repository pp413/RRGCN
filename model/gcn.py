from .utils import *
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class LoopRel(nn.Module):
    
    def __init__(self, num, dim):
        super().__init__()
        self.num = num
        self.dim = dim
        self.loop_rel = get_param((num, dim))
    
    def forward(self, x):
        return torch.cat([x, self.loop_rel], dim=0)
    
    def __repr__(self):
        return 'LoopRel({}, {})'.format(self.num, self.dim)

class MM(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = get_param((in_channels, out_channels))
    
    def forward(self, x):
        return torch.mm(x, self.weight)
    
    def __repr__(self):
        return 'Weight({}, {})'.format(self.in_channels, self.out_channels)

class RelNorm(nn.Module):
    
    def __init__(self, num_rels, dim):
        super().__init__()
        self.num_rels = num_rels
        self.dim = dim
        self.weight = get_param((num_rels, dim))
    
    def forward(self):
        return F.normalize(self.weight, p=2, dim=-1)
    
    def __repr__(self):
        return 'Rel_Norm({}, {})'.format(self.num_rels, self.dim)


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_index, edge_type, act=lambda x: x, params=None):
        super(self.__class__, self).__init__(flow='target_to_source', aggr='add')
        self.edge_index = edge_index
        self.edge_type = edge_type

        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = params.num_rel
        self.num_ents = params.num_ent
        
        self.w_ent = MM(in_channels, out_channels)
        self.w_rel = MM(in_channels, out_channels)
        # self.norm_rel = RelNorm(2 * self.num_rels + 1, in_channels)
        # self.attention = MM(4 * in_channels, 1)
        
        self.loop_rel = LoopRel(1, in_channels)

        self.drop = torch.nn.Dropout(p=params.encoder_gcn_drop)
        self.bn_ent = torch.nn.BatchNorm1d(out_channels)
        # self.bn_rel = torch.nn.BatchNorm1d(out_channels)
        
        self.highway = Highway(out_channels, out_channels)
        self.act = nn.Tanh()

        self.register_parameter(
            'bias', Parameter(torch.zeros(out_channels)))
        
        self.edge_norm = self.compute_norm(self.edge_index, self.num_ents).detach()

    def forward(self, x, edge_index, edge_type, rel_embed):
        rel_embed = self.loop_rel(rel_embed)
        
        if self.p.debug:
            print('edge index:\n', edge_index)
        
        out = self.propagate(self.edge_index,
                             x=x,
                             edge_type=self.edge_type,
                             rel_feature=F.normalize(rel_embed, p=2, dim=-1),
                             edge_norm=self.edge_norm)
        
        if self.p.debug:
            print('...out:\n', out)

        out = self.w_ent(out)
        out = self.bn_ent(out)
        out = self.act(out)
        out = self.highway(x, out)
        
        rel_embed = self.w_rel(rel_embed[:-1, :])
        
        if self.p.encoder_gcn_bias:
            out = out + self.bias
        if self.p.debug:
            print('---out:\n', out)
        return out, rel_embed

    def rel_transform(self, ent_embed, rel_embed):
        
        trans_embed = ent_embed - 2 * torch.sum(ent_embed * rel_embed, dim=1, keepdim=True) * rel_embed
        
        return trans_embed

    def message(self, x_i, x_j, edge_type, rel_feature, edge_norm):
        
        if self.p.debug:
            print('before message random x:', torch.randn(8,))
            print('xj:\n', x_j)
        
        rel_feature = torch.index_select(rel_feature, 0, edge_type)
        out = self.rel_transform(x_j, rel_feature)
        
        out = out * edge_norm.view(-1, 1)
        
        if self.p.debug:
            print('after message random x:', torch.randn(8,))
            print('out:\n', out)

        return out

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        # Summing number of weights of the edges
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)
        deg_inv = deg.pow(-0.5)							# D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

        return norm

    def __repr__(self):
        # return super().__repr__(self)
        
        def _addindent(s_, numSpaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * '  ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s
        
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines
        
        main_str = self._get_name() + "("
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str ++ extra_lines[0]
            else: 
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str