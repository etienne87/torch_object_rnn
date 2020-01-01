from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from core.modules import ConvLayer, SequenceWise, sequence_upsample

class OctConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_funcs, up_func):
        super(OctConv, self).__init__()
        self.convh2h = conv_funcs[1](in_channels[1], out_channels[1], 1)
        self.convl2l = conv_funcs[0](in_channels[0], out_channels[0], 1)
        self.convl2h = conv_funcs[0](out_channels[0], out_channels[1], 1)
        self.convh2l = conv_funcs[1](out_channels[1], out_channels[0], 2)
        self.up = up_func

    def forward(self, x):
        l, h = x
        h = self.convh2h(h)
        l = self.convl2l(l)
        h += self.up(self.convl2h(l), h)
        l += self.convh2l(h)
        return (l, h)

    @classmethod
    def make_conv(cls, cin, cout, f=2):
        cin2 = cin // f
        cout2 = cout // f
        conv1 = lambda x,y,s: SequenceWise(ConvLayer(x, y, stride=s))
        conv2 = lambda x,y,s: SequenceWise(ConvLayer(x, y, stride=s))
        return OctConv((cin, cin2), (cout, cout2), (conv1, conv2), sequence_upsample)


if __name__ == '__main__':
    x = (torch.rand(1, 1, 16, 32, 32), torch.rand(1, 1, 8, 64, 64))
    net = OctConv.make_conv(16, 32)
    y = net(x)
    print(y[0].shape, y[1].shape)