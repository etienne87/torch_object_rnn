from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional as F
from core.utils.opts import time_to_batch, batch_to_time
from core.modules import SequenceWise, ConvRNN
from functools import partial


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels, up_func=ConvRNN):
        super(FeaturePyramidNetwork, self).__init__()

        # up = partial(up_func, kernel_size=3, stride=1, padding=1, dilation=1)
        up = lambda x, y: SequenceWise(nn.Conv2d(x, y, 1, 1, 0))
        skip = lambda x, y: SequenceWise(nn.Conv2d(x, y, 1, 1, 0))


        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = skip(in_channels, out_channels)
            layer_block_module = up(out_channels, out_channels)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def upsample(self, x, feat_shape):
        x, n = time_to_batch(x)
        x = F.interpolate(x, size=feat_shape, mode='nearest')
        x = batch_to_time(x, n)
        return x

    def forward(self, x):
        last_inner = self.inner_blocks[-1](x[-1])
        results = []
        results.append(self.layer_blocks[-1](last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_lateral = inner_block(feature)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = self.upsample(last_inner, feat_shape)
            last_inner = inner_lateral + inner_top_down
            results.append(layer_block(last_inner))

        return results


if __name__ == '__main__':
    t, n, c, h, w = 10, 3, 32, 128, 128
    inputs = []

    x1 = torch.rand(t, n, 32, h>>3, w>>3)
    x2 = torch.rand(t, n, 64, h>>4, w>>4)
    x3 = torch.rand(t, n, 128, h>>5, w>>5)

    x = [x1, x2, x3]

    net = FeaturePyramidNetwork([32, 64, 128], 128)
    out = net(x)

    print([item.shape for item in out])