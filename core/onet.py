from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
from core.modules import ConvLayer
from core.recurrent import *
from core.unet import UNet


class ONet(UNet):
    def __init__(self, channel_list, mode='cat'):
        down, up = partial(ConvLSTMCell, stride=2), partial(ConvLSTMCell, stride=1)
        skip = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)
        resize = lambda x,y: F.interpolate(x, size=y.shape[-2:], mode='nearest')
        super(ONet, self).__init__(channel_list, mode, down, up, skip, resize)
        self.feedback = ConvLayer(self.ups[-1].out_channels, 2 * self.downs[0].out_channels, stride=2)

    def forward(self, x):
        res = super().forward(x)
        tmp = self.feedback(self.ups[-1].prev_h)
        i, g = torch.split(tmp, self.downs[0].out_channels, dim=1)
        self.downs[0].prev_h = self.downs[0].prev_h + (torch.sigmoid(i) * torch.tanh(g))
        return res


if __name__ == '__main__':
    t, n, c, h, w = 10, 3, 64, 32, 32
    x = torch.rand(t, n, c, h, w)

    # net = UNet([3, 32, 64, 128, 64, 32, 16], down, up, skip, mode='sum')
    channel_list = [64] * 3 + [256] * 4
    net = ONet(channel_list, mode='cat')
    # out = net(x[0])
    # print([item.shape for item in out])

    net2 = RNNWise(net)
    out2 = net2(x)
    print([item.shape for item in out2])

