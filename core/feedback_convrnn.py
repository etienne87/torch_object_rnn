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
    def __init__(self, base, levels=4):
        super(Feedback, self).__init__()
        self.resize = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        self.convs = nn.ModuleList()
        self.feedbacks = nn.ModuleList()

        for i in range(levels):
            cin, cout = base * 2**i, base * 2**(i+1)
            self.convs.append(ConvLSTMCell(cin, cout, stride=2))
            if i < levels-1:
                self.feedbacks.append(ConvLayer( base * 2**(i+2), base * 2**(i+1) * 2, stride=1)) #x2 for splitting into i,g


    def forward(self, x):
        out = []
        for conv in self.convs:
            x = conv(x)
            out.append(x)

        #perform all feedbacks (can be pipelined?)
        for l in range(len(self.convs)-1):
            src = self.resize(self.convs[l+1].prev_h, self.convs[l].prev_h.shape[-2:])
            tmp = self.feedbacks[l](src)
            i, g = torch.split(tmp, self.convs[l].out_channels, dim=1)
            self.convs[l].prev_h = self.convs[l].prev_h + (torch.sigmoid(i) * torch.tanh(g))

        return out

if __name__ == '__main__':
    t, n, c, h, w = 10, 3, 8, 32, 32
    x = torch.rand(t, n, c, h, w)

    net = RNNWise(Feedback(8))

    out2 = net(x)
    print([item.shape for item in out2])
