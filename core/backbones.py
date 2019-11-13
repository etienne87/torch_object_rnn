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
            Bottleneck(self.base * 4, self.base * 8, 2),
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


from core import pretrained_backbones as pbb
from core.fpn import FeaturePyramidNetwork

class MobileNetFPN(nn.Module):
    def __init__(self, in_channels=1, out_channels=256):
        super(MobileNetFPN, self).__init__()
        self.base = 16
        self.bb = pbb.MobileNet(in_channels, frozen_stages=3, norm_eval=True)
        self.p6 = ConvLayer(self.bb.out_channel_list[-1], out_channels, stride=2)
        self.p7 = ConvLayer(out_channels, out_channels, stride=2)

        out_channel_list = self.bb.out_channel_list + [out_channels, out_channels]
        self.neck = FeaturePyramidNetwork(out_channel_list, out_channels)
        self.levels = 5
        self.cout = out_channels

    def forward(self, x):
        x, n = time_to_batch(x)
        x1 = self.bb(x)
        p6 = self.p6(x1[-1])
        p7 = self.p7(p6)
        x1 += [p6, p7]
        x1 = [batch_to_time(item, n) for item in x1]
        outs = self.neck(x1)
        sources = [time_to_batch(item)[0] for item in outs][::-1]

        return sources

    def reset(self):
        for name, module in self._modules.items():
            if hasattr(module, "reset"):
                module.reset()


if __name__ == '__main__':
    t, n, c, h, w = 10, 3, 3, 128, 128
    x = torch.rand(t, n, c, h, w)
    net = MobileNetFPN(3)
    out = net(x)
    print([item.shape for item in out])
