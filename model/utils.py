import warnings
import time

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

import sys


def get_param(shape):
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param


class Circular_Padding_chw(torch.nn.Module):

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

class Highway(torch.nn.Module):
    
    def __init__(self, x_hidden, x_out_channels):
        super(Highway, self).__init__()
        self.lin = torch.nn.Linear(x_hidden, x_out_channels)
    
    def forward(self, x1, x2):
        gate = torch.sigmoid(self.lin(x1))
        return torch.mul(gate, x2) + torch.mul(1 - gate, x1)


# class Inv2d(nn.Module):
    
#     def __init__(self, in_channels, out_channels, kernel_size, stride, reduction_ratio=1, group_channels = 1):
#         super(Inv2d, self).__init__()
        
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.group_channels = group_channels
#         self.groups = self.out_channels // self.group_channels
        
#         assert self.group_channels * self.groups == out_channels, "channels % group_channels != 0 or channels < group_channels"
        
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels,
#                       out_channels=out_channels,
#                       kernel_size=1, stride=1),
#             nn.BatchNorm2d(out_channels),
#         )
        
#         self.reduce = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels,
#                       out_channels=out_channels // reduction_ratio,
#                       kernel_size=1, stride=1),
#             nn.BatchNorm2d(out_channels // reduction_ratio),
#         )
#         self.span = nn.Conv2d(in_channels=out_channels // reduction_ratio,
#                               out_channels=kernel_size**2 * self.groups,
#                               kernel_size=1, stride=1)
        
#         if stride > 1:
#             self.avgpool = nn.AvgPool2d(stride, stride)
        
#         self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)
    
#     def forward(self, x):
#         weight = self.span(self.reduce(x if self.stride == 1 else self.avgpool(x)))
#         b, c, h, w = weight.shape
#         weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
#         out = self.unfold(self.conv(x)).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
#         out = (weight * out).sum(dim=3).view(b, self.out_channels, h, w)
#         # print(out.size())
#         return out

class Inv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, reduction_ratio=1, group_channels = 1):
        super(Inv2d, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_channels = group_channels
        self.groups = self.out_channels // self.group_channels
        
        assert self.group_channels * self.groups == out_channels, "channels % group_channels != 0 or channels < group_channels"
        
        self.circular_padding = Circular_Padding_chw(kernel_size // 2)
        
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )
        self.span = nn.Conv2d(in_channels=out_channels,
                              out_channels=kernel_size**2 * self.groups,
                              kernel_size=1, stride=1)
        
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        
        self.unfold = nn.Unfold(kernel_size, 1, 0, stride)
    
    def forward(self, x):
        weight = self.span(self.reduce(x))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
        x = self.circular_padding(x)
        out = self.unfold(self.reduce(x)).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.out_channels, h, w)
        return out
