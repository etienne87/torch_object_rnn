from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
from core.modules import ConvLayer
from core.recurrent import *


class Feedback(nn.Module):
    def __init__(self, channel_list):
        super(Feedback, self).__init__()
        self.resize = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        self.convs = nn.ModuleList()
        self.feedbacks = nn.ModuleList()
        levels = len(channel_list)

        down_list = []
        up_list = []

        for i in range(len(channel_list) - 1):
            down_list.append((channel_list[i], channel_list[i + 1]))

            if i < len(channel_list)-2:
                up_list.append((channel_list[i + 2], channel_list[i + 1]))
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs += [ConvLSTMCell(item[0], item[1], stride=2) for item in down_list]
        self.ups += [ConvLayer(item[0], item[1] * 2) for item in up_list]
        

    def forward(self, x):
        out = []
        for conv in self.downs:
            x = conv(x)
            out.append(x)

        #perform all feedbacks (can be pipelined?)
        for l in range(len(self.downs)-1):  
            src = self.resize(self.downs[l+1].prev_h, self.downs[l].prev_h.shape[-2:])
            tmp = self.ups[l](src)
            i, g = torch.split(tmp, self.downs[l].out_channels, dim=1)
            self.downs[l].prev_h = self.downs[l].prev_h + (torch.sigmoid(i) * torch.tanh(g))

        return out


if __name__ == '__main__':
    t, n, c, h, w = 10, 3, 8, 32, 32
    x = torch.rand(t, n, c, h, w)

    channel_list = [8,16,16,16,16]
    net = RNNWise(Feedback(channel_list))

    out2 = net(x)
    print([item.shape for item in out2])
