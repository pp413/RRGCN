from .Base import *
from .utils import get_param
from .gcn import GCNConv
from .gat import GATConv


class KB_GNN(BaseModel):

    def __init__(self, gnn_name, edge_index, edge_type, params):
        super(KB_GNN, self).__init__(params)
        self.edge_index = edge_index
        self.edge_type = edge_type
        
        loop_index = torch.stack([torch.arange(self.p.num_ent), torch.arange(self.p.num_ent)]).to(edge_index.device)
        loop_type = torch.full((self.p.num_ent,), 2 * self.p.num_rel, dtype=torch.long).to(edge_index.device)
        
        self.edge_index = torch.cat([edge_index, loop_index], dim=1)
        self.edge_type = torch.cat([edge_type, loop_type], dim=0)
        
        self.gcn_conv_list = nn.ModuleList()
        self.encoder_drop1 = nn.Dropout(params.encoder_drop1)
        self.encoder_drop2 = nn.Dropout(params.encoder_drop2)
        hidden_channels = params.gcn_hidden_channels if params.gcn_num_layers > 1 else params.gcn_out_channels
        
        self.bn_ent = torch.nn.BatchNorm1d(num_features=params.gcn_out_channels)
        self.bn_rel = torch.nn.BatchNorm1d(num_features=params.gcn_out_channels)
        
        GNNConv = GCNConv if gnn_name == 'GCN' else GATConv 
        
        self.gcn_conv_list.append(GNNConv(params.gcn_in_channels, hidden_channels,
                                          self.edge_index, self.edge_type, params=params))
        if params.gcn_num_layers > 1:
            for _ in range(0 if params.gcn_num_layers <=2 else params.gcn_num_layers -2):
                self.gcn_conv_list.append(GNNConv(hidden_channels, hidden_channels,
                                                self.edge_index, self.edge_type, params=params))
            self.gcn_conv_list.append(GNNConv(hidden_channels, params.gcn_out_channels,
                                            self.edge_index, self.edge_type, params=params))

    def forward(self, ent_embed, rel_embed):
        
        x = self.bn_ent(ent_embed)
        r = self.bn_rel(rel_embed)
        
        x, r = self.gcn_conv_list[0](x, self.edge_index, self.edge_type, r)
        x = self.encoder_drop1(x)
        
        if self.p.gcn_num_layers == 1:
            return x, r
        
        x, r = self.gcn_conv_list[1](x, self.edge_index, self.edge_type, r)
        x = self.encoder_drop2(x)

        return x, r