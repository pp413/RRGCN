from .Base import *
from .utils import get_param


class Embeddings(BaseModel):
    
    def __init__(self, params):
        super(Embeddings, self).__init__(params)
        self.embed_dim = params.embed_dim
        if self.p.debug:
            print('before ent embeddings random x:', torch.randn(8,))
        self.init_embed = get_param((params.num_ent, params.embed_dim))
        
        if self.p.debug:
            print('after ent embeddings random x:', torch.randn(8,))
        
        if self.p.num_bases > 0:
            self.init_rel = get_param((params.num_rel, params.embed_dim))
            self.num_rel = params.num_rel
        else:
            if self.p.decoder.lower() == 'transe':
                self.init_rel = get_param((params.num_rel, params.embed_dim))
            else:
                self.init_rel = get_param((params.num_rel * 2, params.embed_dim))
            self.num_rel = params.num_rel * 2
    
    def forward(self):
        
        ent_embeddings = self.init_embed
        rel_embeddings = self.init_rel
        if self.p.decoder.lower() == 'transe':
                rel_embeddings = torch.cat([self.init_rel, -self.init_rel], dim=0)
        
        return ent_embeddings, rel_embeddings
    
    def __repr__(self):
        msg = ''
        
        return 'Embeddings(\n'\
            '  (ent_embeddings): Paramsters({}, {}),\n'\
            '  (rel_embeddings): Parameters({}, {})\n'\
            '  )'.format(self.num_ent, self.p.embed_dim, self.num_rel, self.p.embed_dim, msg)