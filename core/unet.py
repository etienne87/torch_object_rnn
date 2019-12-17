from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional as F
from core.utils.opts import time_to_batch, batch_to_time
from core.modules import ConvLayer, SequenceWise, ConvRNN, Bottleneck
from functools import partial


def sequence_upsample(x, y):
    x, n = time_to_batch(x)
    x = F.interpolate(x, size=y.shape[-2:], mode='nearest')
    x = batch_to_time(x, n)
    return x

class UNet(nn.Module):
    def __init__(self, channel_list, mode, down, up, skip, resize):
        """
        UNET generic: user's choice of layers

        :param in_channels:
        :param channel_list: odd list of channels for all layers
        :param mode: 'sum' or 'cat'
        :param down: down function with signature f(x, y), from channels x to y
        :param up: up function with signature f(x, y), from channels x to y
        :param skip: skip function with signature f(x, y), from channels x to y
        :param resize: resize function with signature f(x, y), resize x like y
        """
        super(UNet, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.mode = mode
        self.resize = resize

        self.down_list, self.up_list, self.skip_list = self.get_inout_channels_unet(channel_list, mode)
        
        self.downs += [down(item[0], item[1]) for item in self.down_list]
        self.ups += [up(item[0], item[1]) for item in self.up_list]
        if self.mode == 'sum':
            self.skips = nn.ModuleList()
            self.skips += [skip(item[0], item[1]) for item in self.skip_list]
        else:
            self.skips = [lambda x:x for _ in self.up_list]

    @staticmethod
    def print_shapes(activation_list):
        print([item.shape for item in activation_list])

    @staticmethod
    def get_inout_channels_unet(channel_list, mode):
        assert len(channel_list) % 2 == 1
        encoders = []
        skips = []
        decoders = []
        if mode == 'sum':
            middle = (len(channel_list) - 1) // 2
            for i in range(len(channel_list) - 1):
                if i < middle:
                    encoders.append((channel_list[i], channel_list[i + 1]))
                else:
                    mirror = middle - (i + 1 - middle)
                    skips.append((channel_list[mirror], channel_list[i + 1]))
                    decoders.append((channel_list[i], channel_list[i + 1]))
        else:
            middle = (len(channel_list) - 1) // 2
            for i in range(len(channel_list) - 1):
                if i < middle:
                    encoders.append((channel_list[i], channel_list[i + 1]))
                else:
                    current = channel_list[i]
                    mirror = middle - (i + 1 - middle)
                    remain = channel_list[i + 1] - channel_list[mirror]
                    assert remain > 0, "[concat] make sure outchannels of decoders are bigger than encoders"
                    decoders.append((channel_list[i], remain))
        return encoders, decoders, skips

    def fuse(self, x, y):
        if self.mode == 'cat':
            return torch.cat([x, y], dim=2)
        else:
            return x + y

    def forward(self, x):
        xin = x
        outs = [x]
        for down_layer in self.downs:
            x = down_layer(x)
            outs.append(x)

        middle = len(outs) - 1

        for i, (skip_layer, up_layer) in enumerate(zip(self.skips, self.ups)):
            side = skip_layer(outs[middle - i - 1])
            top = up_layer(self.resize(outs[-1], side))
            x = self.fuse(top, side)
            outs.append(x)

        return outs

    def reset(self, mask):
        for module in self.downs:
            if hasattr(module, "reset"):
                module.reset(mask)

        for module in self.ups:
            if hasattr(module, "reset"):
                module.reset(mask)

    def _repr_module_list(self, module_list):
        repr = ''
        for i, module in enumerate(module_list):
            repr += str(i)+': '+str(module[0])+";"+str(module[1])+'\n'
        return repr

    def __repr__(self):
        repr = ''
        repr += 'downs: ' + '\n'
        repr += self._repr_module_list(self.down_list)
        repr += 'skips: ' + '\n'
        repr += self._repr_module_list(self.skip_list)
        repr += 'ups: ' + '\n'
        repr += self._repr_module_list(self.up_list)
        return repr

    @classmethod
    def recurrent_unet(cls, channel_list, mode='sum'):
        down = lambda x, y: ConvRNN(x, y, stride=2)
        up = lambda x, y: ConvRNN(x, y)
        skip = lambda x, y: SequenceWise(nn.Conv2d(x, y, kernel_size=3, stride=1, padding=1))
        return UNet(channel_list, mode, down, up, skip, sequence_upsample)


if __name__ == '__main__':
    t, n, c, h, w = 10, 3, 64, 32, 32
    x = torch.rand(t, n, c, h, w)

    channel_list_1 = [3, 32, 64, 128, 128, 64, 32]
    channel_list = [64] * 3 + [256] * 4
    net = UNet.recurrent_unet(channel_list, mode='cat')
    print(net)
    out = net(x)
    print([item.shape for item in out])