from .Base import *
from .utils import Inv2d
from torch.nn import functional as F

class Test(nn.Module):
    def __init__(self, n):
        super(Test, self).__init__()
        self.n = n
        
    def forward(self, x):
        print('Num:', self.n, x.size())
        return x

class Circular_Padding_chw(nn.Module):

    def __init__(self, padding):
        super(Circular_Padding_chw, self).__init__()
        self.padding = padding
        print('padding....')

    def forward(self, batch):
        upper_pad = batch[..., -self.padding:, :]
        lower_pad = batch[..., :self.padding, :]
        temp = torch.cat([upper_pad, batch, lower_pad], dim=2)

        left_pad = temp[..., -self.padding:]
        right_pad = temp[..., :self.padding]
        padded = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded

# TransE Class
class TransE(BaseModel):
    
    def forward(self, sub_embed, rel_embed, all_embed):
        obj_embed = sub_embed + rel_embed
        x = self.p.gamma - torch.norm(obj_embed.unsqueeze(1) - all_embed, p=1, dim=2)
        
        return torch.sigmoid(x)
    
    def pair_feature(self, sub_embed, rel_embed):
        return sub_embed + rel_embed


# DistMult Class
class DistMult(BaseModel):
    
    def forward(self, sub_embed, rel_embed, all_embed):
        obj_embed = sub_embed * rel_embed
        x = torch.mm(obj_embed, all_embed.t())
        # x += self.bias.expand_as(x)
        
        return torch.sigmoid(x)

    def pair_feature(self, sub_embed, rel_embed):
        return sub_embed * rel_embed


# ConvE Class
class ConvE(BaseModel):
    
    def __init__(self, params):
        super(ConvE, self).__init__(params)
        
        ker_size_list = [int(x) for x in params.ker_sz.split(',')]
        self.num = len(ker_size_list)
        _tmp_num_split = params.num_filters // self.num
        steps = None
        if params.num_filters % self.num == 0:
            steps = [_tmp_num_split] * self.num
        else:
            steps = [params.num_filters // (self.num - 1)] * (self.num -1) + [params.num_filters % (self.num - 1)]
        
        _step = _tmp_num_split if _tmp_num_split * self.num == params.num_filters else _tmp_num_split + 1
        
        self.convolutions = nn.ModuleList()
        
        for i, k in enumerate(steps):
            i_k_sz = ker_size_list[i]
            
            flat_sz_h = int(2 * self.p.k_w) - i_k_sz + 1
            flat_sz_w = self.p.k_h - i_k_sz + 1
            flat_sz = flat_sz_h * flat_sz_w * k
            
            self.convolutions.append(nn.Sequential(
                nn.BatchNorm2d(1),
                nn.Conv2d(1, out_channels=k, kernel_size=(i_k_sz, i_k_sz), stride=1, padding=0, bias=params.conve_bias),
                nn.BatchNorm2d(k),
                nn.ReLU(),
                nn.Dropout(params.decoder_feat_drop),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(flat_sz, params.gcn_out_channels),
                nn.Dropout(params.decoder_hid_drop),
                nn.BatchNorm1d(num_features=params.gcn_out_channels),
                nn.ReLU(),
            ))

        self.register_parameter('bias', nn.Parameter(torch.zeros(params.num_ent), requires_grad=True))
    
    def concat(self, ent_embed, rel_embed):
        ent_embed = ent_embed.view(-1, 1, self.p.gcn_out_channels)
        rel_embed = rel_embed.view(-1, 1, self.p.gcn_out_channels)
        stack_inp = torch.cat((ent_embed, rel_embed), dim=1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def pair_feature(self, sub_embed, rel_embed):
        stk_inp = self.concat(sub_embed, rel_embed)
        x = []
        for i in range(self.num):
            x.append(self.convolutions[i](stk_inp))
        
        return torch.cat(x, dim=1)
    
    def forward(self, sub_embed, rel_embed, all_embed):
        # print(sub_embed)
        stk_inp = self.concat(sub_embed, rel_embed)
        # print(sub_embed.size(), stk_inp.size())
        
        pred_lists = []
        
        for i in range(self.num):
            x = self.convolutions[i](stk_inp)
            x = torch.mm(x, all_embed.transpose(1, 0))
            pred_lists.append(x)
        
        pred = pred_lists[0]
        
        for i in range(1, self.num):
            pred += pred_lists[i]
        
        pred += self.bias.expand_as(pred)
        score = torch.sigmoid(pred)
        return score


class InteractE(BaseModel):
    
    def __init__(self, params):
        super(InteractE, self).__init__(params)
        
        ker_size_list = [int(x) for x in params.ker_sz.split(',')]
        self.num = len(ker_size_list)
        _tmp_num_split = params.num_filters // self.num
        steps = None
        if params.num_filters % self.num == 0:
            steps = [_tmp_num_split] * self.num
        else:
            steps = [params.num_filters // (self.num - 1)] * (self.num -1) + [params.num_filters % (self.num - 1)]
        
        _step = _tmp_num_split if _tmp_num_split * self.num == params.num_filters else _tmp_num_split + 1
        
        self.convolutions = nn.ModuleList()
        self.cir_padding = []
        
        for i, k in enumerate(steps):
            i_k_sz = ker_size_list[i]
            
            flat_sz = self.p.k_h * 2 * self.p.k_w * k

            self.convolutions.append(nn.Sequential(
                nn.BatchNorm2d(1),
                Circular_Padding_chw(i_k_sz // 2),
                nn.Conv2d(1, out_channels=k, kernel_size=(i_k_sz, i_k_sz), stride=1, padding=0, groups=1, bias=params.conve_bias),
                nn.BatchNorm2d(k),
                nn.ReLU(),
                nn.Dropout(params.decoder_feat_drop),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(flat_sz, params.gcn_out_channels),
                nn.Dropout(params.decoder_hid_drop),
                nn.BatchNorm1d(num_features=params.gcn_out_channels),
                nn.ReLU(),
            ))

        self.register_parameter('bias', nn.Parameter(torch.zeros(params.num_ent), requires_grad=True))
    
    def concat(self, ent_embed, rel_embed):
        ent_embed = ent_embed.view(-1, 1, self.p.gcn_out_channels)
        rel_embed = rel_embed.view(-1, 1, self.p.gcn_out_channels)
        stack_inp = torch.cat((ent_embed, rel_embed), dim=1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))

        return stack_inp

    def pair_feature(self, sub_embed, rel_embed):
        stk_inp = self.concat(sub_embed, rel_embed)
        x = []
        for i in range(self.num):
            x.append(self.convolutions[i](stk_inp))
        
        return torch.cat(x, dim=1)
    
    def forward(self, sub_embed, rel_embed, all_embed):
        # print(sub_embed)
        stk_inp = self.concat(sub_embed, rel_embed)
        if self.p.debug:
            print(sub_embed.size(), stk_inp.size())
        
        pred_lists = []
        
        for i in range(self.num):
            x = self.convolutions[i](stk_inp)
            pred_i = torch.mm(x, all_embed.transpose(1, 0))
            pred_lists.append(pred_i)
        
        pred = pred_lists[0]
        
        for i in range(1, self.num):
            pred += pred_lists[i]
        
        pred += self.bias.expand_as(pred)
        score = torch.sigmoid(pred)
        return score


class Involuation(BaseModel):
    
    def __init__(self, params):
        super(Involuation, self).__init__(params)
        
        ker_size_list = [int(x) for x in params.ker_sz.split(',')]
        self.num = len(ker_size_list)
        _tmp_num_split = params.num_filters // self.num
        steps = None
        if params.num_filters % self.num == 0:
            steps = [_tmp_num_split] * self.num
        else:
            steps = [params.num_filters // (self.num - 1)] * (self.num -1) + [params.num_filters % (self.num - 1)]
        
        _step = _tmp_num_split if _tmp_num_split * self.num == params.num_filters else _tmp_num_split + 1
        
        self.convolutions = nn.ModuleList()
        
        for i, k in enumerate(steps):
            i_k_sz = ker_size_list[i]
            
            flat_sz_h = int(2 * self.p.k_w) - i_k_sz + 1
            flat_sz_w = self.p.k_h - i_k_sz + 1
            flat_sz = flat_sz_h * flat_sz_w * k
            
            self.convolutions.append(nn.Sequential(
                nn.BatchNorm2d(1),
                nn.Conv2d(1, out_channels=k, kernel_size=(i_k_sz, i_k_sz), stride=1, padding=0, bias=params.conve_bias),
                nn.BatchNorm2d(k),
                nn.ReLU(),
                nn.Dropout(params.decoder_feat_drop),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(flat_sz, params.gcn_out_channels),
                nn.Dropout(params.decoder_hid_drop),
                nn.BatchNorm1d(num_features=params.gcn_out_channels),
                nn.ReLU(),
            ))

        self.register_parameter('bias', nn.Parameter(torch.zeros(params.num_ent), requires_grad=True))
        
        self.batch_features = None
    
    def concat(self, ent_embed, rel_embed):
        ent_embed = ent_embed.view(-1, 1, self.p.gcn_out_channels)
        rel_embed = rel_embed.view(-1, 1, self.p.gcn_out_channels)
        stack_inp = torch.cat((ent_embed, rel_embed), dim=1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def pair_feature(self, sub_embed, rel_embed):
        stk_inp = self.concat(sub_embed, rel_embed)
        x = []
        for i in range(self.num):
            x.append(self.convolutions[i](stk_inp))
        
        return torch.cat(x, dim=1)
    
    def forward(self, sub_embed, rel_embed, all_embed):
        # print(sub_embed)
        stk_inp = self.concat(sub_embed, rel_embed)
        # print(sub_embed.size(), stk_inp.size())
        
        pred_lists = []
        
        for i in range(self.num):
            x = self.convolutions[i](stk_inp)
            x = torch.mm(x, all_embed.transpose(1, 0))
            pred_lists.append(x)
        
        pred = pred_lists[0]
        
        for i in range(1, self.num):
            pred += pred_lists[i]
        
        pred += self.bias.expand_as(pred)
        score = torch.sigmoid(pred)
        return score
        
        