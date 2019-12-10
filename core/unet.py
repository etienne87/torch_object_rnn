from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional as F
from core.utils.opts import time_to_batch, batch_to_time
from core.modules import ConvLayer, SequenceWise, ConvLSTM, Bottleneck, PyramidPooling
from functools import partial


class UNet(nn.Module):
    def __init__(self, channel_list, mode='sum', stride=2, down_func=ConvLSTM, up_func=ConvLSTM):
        """
        UNET generic

        :param in_channels:
        :param channel_list: odd list of channels for all layers
        :param mode: 'sum' or 'cat'
        :param stride: multiple of 2
        :param down_func: down function
        :param up_func: up function
        :param skip_func: skip function
        """
        super(UNet, self).__init__()

        down = partial(down_func, kernel_size=3, stride=stride, dilation=1)
        up = partial(up_func, kernel_size=3, stride=1, dilation=1)
        skip = lambda x, y: SequenceWise(PyramidPooling(), nn.Conv2d(x*4, y, 1, 1, 0))
        # skip = lambda x, y: SequenceWise(nn.Conv2d(x, y, 1, 1, 0))

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        self.downstride = stride
        self.mode = mode

        down_list, up_list, skip_list = self.get_inout_channels_unet(channel_list, mode)

        self.downs += [down(item[0], item[1]) for item in down_list]
        self.ups += [up(item[0], item[1]) for item in up_list]
        if self.mode == 'sum':
            self.skips = nn.ModuleList()
            self.skips += [skip(item[0], item[1]) for item in skip_list]
        else:
            self.skips = [lambda x:x for _ in up_list][:-1]
        
        self.down_list = down_list
        self.up_list = up_list 
        self.skip_list = skip_list

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

            skips = skips[:-1]
        else:
            middle = (len(channel_list) - 1) // 2
            for i in range(len(channel_list) - 1):
                if i < middle:
                    encoders.append((channel_list[i], channel_list[i + 1]))
                elif i < len(channel_list) - 2:
                    mirror = middle - (i + 1 - middle)
                    remain = channel_list[i + 1] - mirror
                    assert remain > 0, "[concat] make sure outchannels of decoders are bigger than encoders"
                    decoders.append((channel_list[i], remain))
                else:
                    decoders.append((channel_list[i], channel_list[i + 1]))

        return encoders, decoders, skips

    def fuse(self, x, y):
        if self.mode == 'cat':
            return torch.cat([x, y], dim=2)
        else:
            return x + y

    def upsample(self, x):
        x, n = time_to_batch(x)
        x = F.interpolate(x, scale_factor=self.downstride, mode='nearest')
        x = batch_to_time(x, n)
        return x

    def forward(self, x):
        outs = []
        for down_layer in self.downs:
            x = down_layer(x)
            outs.append(x)

        middle = len(outs)-1

        for i, (skip_layer, up_layer) in enumerate(zip(self.skips, self.ups)):
            top = up_layer(self.upsample(outs[-1]))
            side = skip_layer(outs[middle - i - 1])
            # print('side: ', side.shape, ' top: ', top.shape)
            x = self.fuse(top, side)
            outs.append(x)

        x = self.ups[-1](self.upsample(outs[-1]))
        outs.append(x)

        return outs

    def reset(self, mask=None):
        for module in self.downs:
            if hasattr(module, "reset"):
                module.reset()

        for module in self.ups:
            if hasattr(module, "reset"):
                module.reset()

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



if __name__ == '__main__':
    t, n, c, h, w = 10, 3, 3, 128, 128
    x = torch.rand(t, n, c, h, w)
    net = UNet([3, 32, 64, 128, 64, 32, 16], mode='sum')
    out = net(x)
