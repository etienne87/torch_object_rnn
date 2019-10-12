from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from torch.nn import functional as F
import torch
from core.utils.opts import time_to_batch, batch_to_time


def get_padding(kernel, dilation):
    k2 = kernel + (kernel-1) * (dilation-1)
    return k2//2


class ParallelWise(nn.Sequential):
    def __init__(self, *args):
        super(ParallelWise, self).__init__(*args)

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = time_to_batch(x)[0]
        x = super().forward(x)
        x = batch_to_time(x, n)
        return x


class SequenceWise(nn.Sequential):
    def __init__(self, *args):
        super(SequenceWise, self).__init__(*args)

    def forward(self, x):
        xiseq = x.split(1, 0)
        res = []
        for t, xt in enumerate(xiseq):
            y = super().forward(xt[0])
            res.append(y.unsqueeze(0))
        return torch.cat(res, dim=0)


class SeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                      bias=bias),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        )


class ConvLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, dilation=1,
                 bias=True, norm='InstanceNorm2d', activation='LeakyReLU', separable=False):

        conv_func = SeparableConv2d if separable else nn.Conv2d
        super(ConvLayer, self).__init__(
            conv_func(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=padding, bias=bias),
            nn.Identity() if norm == 'none' else getattr(nn, norm)(out_channels),
            getattr(nn, activation)()
        )


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        mid_planes = planes//4
        self.conv1 = ConvLayer(in_channels=in_planes, out_channels=mid_planes, kernel_size=1, padding=0, bias=False)
        self.conv2 = ConvLayer(in_channels=mid_planes, out_channels=mid_planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.conv3 = ConvLayer(in_channels=mid_planes, out_channels=planes, kernel_size=1, padding=0, bias=False)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.downsample = ConvLayer(in_channels=in_planes, out_channels=planes,
                                     kernel_size=1, padding=0, stride=stride,
                          bias=False, activation='Identity')


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[3, 6, 12, 18]):
        super(ASPP, self).__init__()
        modules = []
        for rate in atrous_rates:
            modules.append(ConvLayer(in_channels, out_channels, dilation=rate, padding=rate))

        self.convs = nn.ModuleList(modules)
        self.project = ConvLayer(out_channels * len(self.convs), out_channels)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res = self.project(res)
        return res




class RNN(nn.Module):
    """
    base class that has memory, each class with hidden state has to derive from basernn
    TODO: add the saturation cost as a hook!!

    """

    def __init__(self, hard):
        super(RNN, self).__init__()
        self.saturation_cost = 0
        self.saturation_limit = 1.05
        self.saturation_weight = 1e-1
        self.set_gates(hard)

    def set_gates(self, hard):
        if hard:
            self.sigmoid = self.hard_sigmoid
            self.tanh = self.hard_tanh
        else:
            self.sigmoid = torch.sigmoid
            self.tanh = torch.tanh

    def hard_sigmoid(self, x_in):
        self.add_saturation_cost(x_in)
        x = x_in * 0.5 + 0.5
        y = torch.clamp(x, 0.0, 1.0)
        return y

    def hard_tanh(self, x):
        self.add_saturation_cost(x)
        y = torch.clamp(x, -1.0, 1.0)
        return y

    def add_saturation_cost(self, var):
        """Calculate saturation cost."""
        sat_loss = F.relu(torch.abs(var) - self.saturation_limit)
        cost = sat_loss.mean()

        cost *= self.saturation_weight
        self.saturation_cost += cost

    def reset(self):
        raise NotImplementedError()

    def detach(self):
        raise NotImplementedError()


class ConvLSTM(RNN):
    r"""ConvLSTMCell module, applies sequential part of LSTM.
    """

    def __init__(self, in_channels, hidden_dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, conv_func=nn.Conv2d, hard=False):
        super(ConvLSTM, self).__init__(hard)
        self.hidden_dim = hidden_dim

        self.conv_h2h = conv_func(in_channels=in_channels + self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  bias=bias)

        self.reset()

    def forward(self, x):
        if self.prev_h is None:
            height, width = x.shape[-2:]
            self.prev_h = torch.zeros((x.size(0), self.hidden_dim, height, width), dtype=torch.float32, device=x.device)
            self.prev_c = torch.zeros((x.size(0), self.hidden_dim, height, width), dtype=torch.float32, device=x.device)

        xh = torch.cat([x,self.prev_h], dim=1)
        tmp = self.conv_h2h(xh)

        cc_i, cc_f, cc_o, cc_g = torch.split(tmp, self.hidden_dim, dim=1)
        i = self.sigmoid(cc_i)
        f = self.sigmoid(cc_f)
        o = self.sigmoid(cc_o)
        g = self.tanh(cc_g)
        c = f * self.prev_c + i * g
        h = o * self.tanh(c)
        self.prev_c = c
        self.prev_h = h
        return h

    def detach(self):
        if self.prev_h is not None:
            self.prev_h = self.prev_h.detach()
            self.prev_c = self.prev_c.detach()

    def reset(self):
        self.prev_h, self.prev_c = None, None




if __name__ == '__main__':
    x = torch.rand(3, 5, 128, 16, 16)

    net = SequenceWise(ConvLSTM(128, 256))
    #net = ParallelWise(nn.Conv2d(128, 128, 1, 1, 0), nn.ReLU())


    y = net(x)
    print(y.shape)