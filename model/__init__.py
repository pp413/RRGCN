from .Base import *
from .embeddings import Embeddings
from .encoder import KB_GNN
from .decoder import TransE, DistMult, ConvE, InteractE, Involuation



__all__ = ['Embed_TransE', 'Embed_DistMult', 'Embed_ConvE', 'Embed_InteractE', 'Embed_Involuation',
           'GCN_TransE', 'GCN_DistMult', 'GCN_ConvE', 'GCN_InteractE',
           'GAT_ConvE']


# ############################################################################################
#                 Models
# ############################################################################################

class Model(BaseModel):
    
    def __init__(self, params, edge_index, edge_type):
        super(Model, self).__init__(params)
        
        self.edge_index = edge_index.long()
        self.edge_type = edge_type.long()
        
        if self.p.debug:
            print('before embeddings random x:', torch.randn(8))
        
        self.create_embeddings()
        if self.p.debug:
            print('after embeddings random x:', torch.randn(8))
        self.create_encoder()
        self.create_decoder()
        
        self.bce_loss = nn.BCELoss()
        
        dim = self.p.embed_dim if self.encoder is None else self.p.gcn_out_channels
        
        self.final_ent_embeddings = Parameter(torch.empty(self.num_ent, dim), requires_grad=False)
        self.final_rel_embeddings = Parameter(torch.empty(2 * self.num_rel, dim), requires_grad=False)
        
        self.global_loss = None
    
    @property
    def device(self):
        return self.final_ent_embeddings.data.device
    
    @device.setter
    def device(self, str_device):
        self.to(str_device)
    
    def loss(self, batch_data, label):
        
        pred, rel_embed, all_embed = self.forward(batch_data)
        
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label)
        label = label.to(self.device)
        
        if self.p.debug:
            print(pred.size(), label.size())
        
        if isinstance(pred, tuple):
            return self.bce_loss(pred[0], label) + self.bce_loss(pred[2], label) + self.bce_loss(pred[2], label)
        else:
            return self.bce_loss(pred, label)
    
    def look_up(self, sub, rel, ent_embeddings, rel_embeddings):
        sub_embed = torch.index_select(ent_embeddings, 0, sub)
        rel_embed = torch.index_select(rel_embeddings, 0, rel)
        # obj_embed = ent_embeddings
        return sub_embed, rel_embed
    
    def create_embeddings(self):
        self.embeddings = None

    def create_encoder(self):
        self.encoder = None
    
    def create_decoder(self):
        self.decoder = None
    
    def gnn_forward(self, ent_embeddings, rel_embeddings):
        
        if self.p.debug:
            print('init_embeddings:\n', ent_embeddings)
            print('init_rel:\n', rel_embeddings)
            print('random x:', torch.randn(8,))
        
        if self.encoder is not None:
            return self.encoder(ent_embeddings, rel_embeddings)
        else:
            return ent_embeddings, rel_embeddings
    
    def forward(self, batch_data):
        if not isinstance(batch_data, torch.Tensor):
            batch_data = torch.from_numpy(batch_data)
        batch_data = batch_data.to(self.device)
        sub, rel = batch_data[:, 0], batch_data[:, 1]
        
        ent_embeddings, rel_embeddings = self.gnn_forward(*self.embeddings())
        
        if self.p.debug:
            print('X1:\n', ent_embeddings)
            print('R1:\n', rel_embeddings)
            print('random x:', torch.randn(8,))
        sub_embed, rel_embed= self.look_up(sub, rel, ent_embeddings, rel_embeddings)
        
        if self.p.debug:
            print('sub_embed:\n', sub_embed)
            print('rel_embed:\n', rel_embed)
            print('random x:', torch.randn(8,))
        
        features = self.decoder(sub_embed, rel_embed, ent_embeddings)
        return features, rel_embed, ent_embeddings

    def save_embeddings(self):
        self.eval()
        with torch.no_grad():
            ent_embeddings, rel_embeddings = self.gnn_forward(*self.embeddings())
            
            self.final_ent_embeddings.data = ent_embeddings
            self.final_rel_embeddings.data = rel_embeddings
            
            if self.p.debug:
                print('final ent:', self.final_ent_embeddings)
                print('final rel:', self.final_rel_embeddings)
        
        self.train()

    def predict(self, batch_data):
        if not isinstance(batch_data, torch.Tensor):
            batch_data = torch.from_numpy(batch_data)
        batch_data = batch_data.to(self.device)
        sub, rel = batch_data[:, 0], batch_data[:, 1]
        with torch.no_grad():
            ent_embeddings, rel_embeddings = self.final_ent_embeddings.data, self.final_rel_embeddings.data
            sub_embed, rel_embed = self.look_up(sub, rel, ent_embeddings, rel_embeddings)
            features = self.decoder(sub_embed, rel_embed, ent_embeddings)
            if isinstance(features, tuple):
                return features[0]
            else:
                return features
    
    def get_pair_feature(self, batch_data):
        if not isinstance(batch_data, torch.Tensor):
            batch_data = torch.from_numpy(batch_data)
        batch_data = batch_data.to(self.device)
        sub, rel = batch_data[:, 0], batch_data[:, 1]
        with torch.no_grad():
            ent_embeddings, rel_embeddings = self.final_ent_embeddings.data, self.final_rel_embeddings.data
            sub_embed, rel_embed = self.look_up(sub, rel, ent_embeddings, rel_embeddings)
            return self.decoder.pair_feature(sub_embed, rel_embed).cpu().numpy()


class Embed_TransE(Model):
    
    def create_embeddings(self):
        self.embeddings = Embeddings(self.p)
    
    def create_decoder(self):
        self.decoder = TransE(self.p)


class Embed_DistMult(Embed_TransE):
    
    def create_decoder(self):
        self.decoder = DistMult(self.p)
    


class Embed_ConvE(Embed_TransE):
    
    def create_decoder(self):
        self.decoder = ConvE(self.p)


class Embed_InteractE(Embed_TransE):
    
    def create_decoder(self):
        self.decoder = InteractE(self.p)


class Embed_Involuation(Embed_TransE):
    
    def create_decoder(self):
        self.decoder = Involuation(self.p)


class GCN_TransE(Embed_TransE):
    
    def create_encoder(self):
        self.encoder = KB_GNN('GCN', self.edge_index, self.edge_type, self.p)


class GCN_DistMult(Embed_DistMult):
    
    def create_encoder(self):
        self.encoder = KB_GNN('GCN', self.edge_index, self.edge_type, self.p)


class GCN_ConvE(Embed_ConvE):
    
    def create_encoder(self):
        self.encoder = KB_GNN('GCN', self.edge_index, self.edge_type, self.p)
        
class GCN_InteractE(Embed_InteractE):
    
    def create_encoder(self):
        self.encoder = KB_GNN('GCN', self.edge_index, self.edge_type, self.p)

class GAT_ConvE(Embed_ConvE):
    
    def create_encoder(self):
        self.encoder = KB_GNN('GAT', self.edge_index, self.edge_type, self.p)


