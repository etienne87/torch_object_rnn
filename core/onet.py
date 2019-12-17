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
    def __init__(self, channel_list, mode='sum', stride=2):
        down, up = partial(ConvLSTMCell, stride=2), partial(ConvLSTMCell, stride=1)
        skip = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)
        resize = lambda x,y: F.interpolate(x, size=y.shape[-2:], mode='nearest')
        super(ONet, self).__init__(channel_list, mode, down, up, skip, resize)

        self.feedbacks = nn.ModuleList()
        for down, up in zip(self.downs, self.ups[::-1]):
            in_channels = up.out_channels
            out_channels = down.out_channels
            self.feedbacks.append(nn.Conv2d(in_channels, 2 * out_channels, kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        res = super().forward(x)

        for down, up, fb in zip(self.downs, self.ups[::-1], self.feedbacks):
            tmp = fb(up.prev_h)
            i, g = torch.split(tmp, down.out_channels, dim=1)
            if down.prev_h is not None:
                down.prev_h = down.prev_h + torch.sigmoid(i) * torch.tanh(g)

        return res



if __name__ == '__main__':

    t, n, c, h, w = 10, 3, 3, 128, 128
    x = torch.rand(t, n, c, h, w)

    # net = UNet([3, 32, 64, 128, 64, 32, 16], down, up, skip, mode='sum')
    net = ONet([3, 32, 64, 128, 128, 64, 32], mode='sum')
    # out = net(x[0])
    # print([item.shape for item in out])

    net2 = RNNWise(net)
    out2 = net2(x)
    print([item.shape for item in out2])


