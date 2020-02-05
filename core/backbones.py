from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from core.utils.opts import time_to_batch, batch_to_time, cuda_time

from core.modules import ConvLayer, SequenceWise, ConvRNN, Bottleneck, BottleneckLSTM, PreActBlock
from core.unet import UNet

from core.recurrent import RNNWise
from core.onet import ONet

from core.feedback_convrnn import Feedback


def ff_preact_stem(cin, base):
    return SequenceWise(nn.Sequential(
            ConvLayer(cin, base * 2, kernel_size=7, stride=2, padding=3),
            PreActBlock(base * 2, base * 4, stride=2),
            PreActBlock(base * 4, base * 8, stride=2),
        ))

class Vanilla(nn.Module):
    def __init__(self, cin=1, cout=256, nmaps=3):
        super(Vanilla, self).__init__()
        self.cin = cin
        self.base = 4
        self.cout = cout
        self.nmaps = nmaps
        self.levels = 2

        self.conv1 = ff_preact_stem(cin, self.base)
       
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2))
        for i in range(self.levels-1):
            m = nn.Sequential(
                ConvRNN(cout, cout, stride=1),
                SequenceWise(PreActBlock(cout, cout, stride=2))
                )
            self.conv2.append(m)

    def forward(self, x):
        x = self.conv1(x)
        outs = []
        for conv in self.conv2:
            x = conv(x)
            outs.append(time_to_batch(x)[0])

        return outs

    def reset(self, mask=None):
        for name, module in self._modules.items():
            if hasattr(module, "reset"):
                module.reset(mask)


class FBN(nn.Module):
    def __init__(self, cin=1, cout=256, nmaps=3):
        super(FBN, self).__init__()
        self.cin = cin
        self.base = 8
        self.cout = cout
        self.nmaps = nmaps
        self.levels = 4

        self.conv1 = ff_preact_stem(cin, self.base)
        self.conv2 = RNNWise(Feedback([self.base * 8, cout, cout, cout, cout]))
        
    def forward(self, x):
        x1 = self.conv1(x)
        outs = self.conv2(x1)
        outs = [time_to_batch(item)[0] for item in outs]
        return outs

    def reset(self, mask=None):
        for name, module in self._modules.items():
            if hasattr(module, "reset"):
                module.reset(mask)
            if hasattr(module, "reset_modules"):
                module.reset_modules(mask)



class FPN(nn.Module):
    def __init__(self, cin=1, cout=256, nmaps=3):
        super(FPN, self).__init__()
        self.cin = cin
        self.base = 8
        self.cout = cout
        self.nmaps = nmaps
        self.levels = 4

        self.conv1 = ff_preact_stem(cin, self.base)  

        self.conv2 = UNet.recurrent_unet([self.base * 8] * (self.levels-1) + [cout] * self.levels, mode='cat')
        #self.conv2 = RNNWise(ONet([self.base * 8] * (self.levels-1) + [cout] * self.levels))

    def forward(self, x):
        x1 = self.conv1(x)
        outs = self.conv2(x1)[-self.levels:]
        sources = [time_to_batch(item)[0] for item in outs][::-1]

        return sources

    def reset(self, mask=None):
        for name, module in self._modules.items():
            if hasattr(module, "reset"):
                module.reset(mask)
            if hasattr(module, "reset_modules"):
                module.reset_modules(mask)


class Trident(nn.Module):
    def __init__(self, cin=1):
        super(Trident, self).__init__()
        self.cin = cin
        base = 8
        self.conv1 = SequenceWise(nn.Sequential(
            Bottleneck(cin, base, 2),
            Bottleneck(base, base * 4, 2),
            Bottleneck(base * 4, base * 8, 2)
        ))
        self.levels = 4

        self.conv3 = ConvRNN(base * 8, base * 8, kernel_size=7, stride=2, padding=3)
        self.conv4 = ConvRNN(base * 8, base * 16, kernel_size=7, stride=2, dilation=1, padding=3)
        self.conv5 = ConvRNN(base * 16, base * 16, kernel_size=7, stride=2, dilation=1, padding=3)
        self.conv6 = ConvRNN(base * 16, base * 16, kernel_size=7, stride=2, dilation=1, padding=3)

        self.out_channel_list = [self.conv3.out_channels,  # 8
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

    def reset(self, mask=None):
        for name, module in self._modules.items():
            if hasattr(module, "reset"):
                module.reset(mask)


from core import pretrained_backbones as pbb
from core.fpn import FeaturePyramidNetwork


class BackboneWithFPN(nn.Module):
    """Backbone with or without FPN"""
    def __init__(self, backbone, out_channels=256):
        super(BackboneWithFPN, self).__init__()
        self.bb = backbone
        self.neck = FeaturePyramidNetwork(self.bb.out_channel_list, out_channels)
        self.levels = 5
        self.cout = out_channels

    def forward(self, x):
        x1 = self.bb(x)
        outs = self.neck(x1)
        sources = [time_to_batch(item)[0] for item in outs]
        return sources

    def reset(self, mask=None):
        for name, module in self._modules.items():
            if hasattr(module, "reset"):
                module.reset(mask)


class MobileNetFPN(BackboneWithFPN):
    def __init__(self, in_channels=3, out_channels=256):
        super(MobileNetFPN, self).__init__(
            pbb.MobileNet(in_channels, frozen_stages=1, norm_eval=True)
        )

class ResNet50FPN(BackboneWithFPN):
    def __init__(self, in_channels=3, out_channels=256):
        super(ResNet50FPN, self).__init__(
            pbb.resnet50(in_channels, True, frozen_stages=-1, norm_eval=True)
        )


class BackboneWithP6P7(nn.Module):
    """Backbone for SSD"""
    def __init__(self, backbone, out_channels=256, add_p6p7=True):
        super(BackboneWithP6P7, self).__init__()
        self.bb = backbone
        self.levels = 5
        self.add_p6p7 = add_p6p7
        if add_p6p7:
            self.p6 = SequenceWise(ConvLayer(backbone.out_channel_list[-1], out_channels, norm='none', stride=2))
            self.p7 = SequenceWise(ConvLayer(out_channels, out_channels, stride=2, norm='none',  activation='Identity'))
        self.out_channel_list = backbone.out_channel_list + [out_channels, out_channels]

    def forward(self, x):
        x1 = self.bb(x)
        p6 = self.p6(x1[-1])
        p7 = self.p7(p6)
        outs = x1 + [p6, p7]
        sources = [time_to_batch(item)[0] for item in outs]
        return sources

    def reset(self, mask=None):
        for name, module in self._modules.items():
            if hasattr(module, "reset"):
                module.reset(mask)


class MobileNetSSD(BackboneWithP6P7):
    def __init__(self, in_channels=3, out_channels=256):
        super(MobileNetSSD, self).__init__(
            pbb.MobileNet(in_channels, True, frozen_stages=1, norm_eval=True)
        )

class ResNet50SSD(BackboneWithP6P7):
    def __init__(self, in_channels=3, out_channels=256):
        super(ResNet50SSD, self).__init__(
            pbb.resnet50(in_channels, True, frozen_stages=-1, norm_eval=True)
        )


if __name__ == '__main__':
    t, n, c, h, w = 10, 3, 3, 128, 128
    x = torch.rand(t, n, c, h, w)
    #net = MobileNetFPN(3)
    net = FPN(3)

    x = x.cuda()
    net.cuda()

    for i in range(10):
        out = net(x)
    print([item.shape for item in out])
