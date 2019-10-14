from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from core.utils.opts import time_to_batch, batch_to_time
from core.modules import ConvLayer, SequenceWise, ConvRNN, Bottleneck, BottleneckLSTM
from core.unet import UNet





class FPN(nn.Module):
    def __init__(self, cin=1, cout=256, nmaps=3):
        super(FPN, self).__init__()
        self.cin = cin
        self.base = 16
        self.cout = cout
        self.nmaps = nmaps

        self.conv1 = SequenceWise(nn.Sequential(
            Bottleneck(cin, self.base * 2, 2),
            Bottleneck(self.base * 2, self.base * 4, 2),
            Bottleneck(self.base * 4, self.base * 8, 2)
        ))

        self.levels = 4
        self.conv2 = UNet([self.base * 8] * (self.levels-1) + [cout] * self.levels)


    def forward(self, x):
        x1 = self.conv1(x)
        outs = self.conv2(x1)[-self.levels:]

        sources = [time_to_batch(item)[0] for item in outs][::-1]

        return sources

    def reset(self):
        for name, module in self._modules.items():
            if hasattr(module, "reset"):
                module.reset()


class Trident(nn.Module):
    def __init__(self, cin=1):
        super(Trident, self).__init__()
        self.cin = cin
        base = 8
        self.conv1 = SequenceWise(nn.Sequential(
            Bottleneck(cin, base, kernel_size=7, stride=2, padding=3),
            Bottleneck(base, base * 4, kernel_size=7, stride=2, padding=3),
            Bottleneck(base * 4, base * 8, kernel_size=7, stride=2, padding=3)
        ))

        self.conv3 = ConvRNN(base * 8, base * 8, kernel_size=7, stride=2, padding=3)
        self.conv4 = ConvRNN(base * 8, base * 16, kernel_size=7, stride=1, dilation=1, padding=3)
        self.conv5 = ConvRNN(base * 16, base * 16, kernel_size=7, stride=1, dilation=2, padding=3)
        self.conv6 = ConvRNN(base * 16, base * 16, kernel_size=7, stride=1, dilation=3, padding=3 * 2)

        self.end_point_channels = [self.conv3.out_channels,  # 8
                                   self.conv4.out_channels,  # 16
                                   self.conv5.out_channels,  # 32
                                   self.conv6.out_channels]  # 64

    def forward(self, x):
        sources = list()

        x2 = self.conv1(x)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        sources = [time_to_batch(item)[0] for item in [x3, x4, x5, x6]]

        return sources

    def reset(self):
        for name, module in self._modules.items():
            if isinstance(module, ConvRNN):
                module.timepool.reset()