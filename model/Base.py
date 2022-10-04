import torch
import torch.nn as nn
from torch.nn import Parameter


class BaseModel(nn.Module):
    
    def __init__(self, params):
        super(BaseModel, self).__init__()
        self.p = params
        self.num_ent = params.num_ent
        self.num_rel = params.num_rel








