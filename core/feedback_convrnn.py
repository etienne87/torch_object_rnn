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
        self.resize = lambda x, size: F.interpolate(x, size=size, mode='bilinear')
        self.convs = nn.ModuleList()
        self.feedbacks = nn.ModuleList()
        levels = len(channel_list)

        down_list = []
        up_list = []
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i in range(len(channel_list) - 1):
            down_list.append((channel_list[i], channel_list[i + 1]))

            if i < len(channel_list)-2:
                self.downs.append(ConvLSTMCell(channel_list[i], channel_list[i + 1], stride=2, feedback_channels=channel_list[i + 2]))
                up_list.append((channel_list[i + 2], channel_list[i + 1]))
            else:
                self.downs.append(ConvLSTMCell(channel_list[i], channel_list[i + 1], stride=2))

    def forward(self, x):
        out = []
        for conv in self.downs:
            x = conv(x)
            out.append(x)

        #perform all feedbacks (can be pipelined?)
        for l in range(len(self.downs)-1):  
            self.downs[l].prev_fb = self.resize(self.downs[l+1].prev_h, self.downs[l].prev_h.shape[-2:])
        
        return out


if __name__ == '__main__':
    t, n, c, h, w = 10, 3, 8, 32, 32
    x = torch.rand(t, n, c, h, w)

    channel_list = [8,16,32,64,16]
    net = RNNWise(Feedback(channel_list))

    out2 = net(x)
    print([item.shape for item in out2])

    y = out2[-1].sum()

    y.backward()

    import pdb;pdb.set_trace()
    print('test')
