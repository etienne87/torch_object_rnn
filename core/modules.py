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


class SequenceWise(nn.Sequential):
    def __init__(self, *args, parallel=True):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        If RNN (has reset module), will bypass reshapes
        :param module: Module to apply input to.
        :param parallel: Run in Parallel, or Sequentially. Not equivalent for BatchNorm for instance.
        """
        super(SequenceWise, self).__init__(*args)
        self.parallel = parallel

    def forward_parallel(self, x):
        t, n = x.size(0), x.size(1)
        x = time_to_batch(x)[0]
        x = super().forward(x)
        x = batch_to_time(x, n)
        return x

    def forward_sequential(self, x):
        xiseq = x.split(1, 0)
        res = []
        for t, xt in enumerate(xiseq):
            y = super().forward(xt[0])
            res.append(y.unsqueeze(0))
        return torch.cat(res, dim=0)

    def forward(self, x):
        if x.dim() == 4:
            return super().forward(x)
        elif self.parallel:
            return self.forward_parallel(x)
        else:
            return self.forward_sequential(x)


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
                 bias=True, norm='InstanceNorm2d', activation='LeakyReLU', separable=False, norm_before_conv=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        conv_func = SeparableConv2d if separable else nn.Conv2d
        if norm_before_conv:
            super(ConvLayer, self).__init__(
            nn.Identity() if norm == 'none' else getattr(nn, norm)(in_channels),
            conv_func(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=padding, bias=bias),
            getattr(nn, activation)()
        )
        else:
            super(ConvLayer, self).__init__(
                conv_func(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                        padding=padding, bias=bias),
                nn.Identity() if norm == 'none' else getattr(nn, norm)(out_channels),
                getattr(nn, activation)()
            )


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.in_channels = in_planes 
        self.out_channels = planes
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

class PyramidPooling(nn.Module):
    def __init__(self, kernel_sizes=[1,3,5,7]):
        super(PyramidPooling, self).__init__()
        self.pools = nn.ModuleList()
        for k in kernel_sizes:
            self.pools.append(nn.AvgPool2d(kernel_size=k, stride=1, padding=k//2))
    
    def forward(self, x):
        y = []
        for pool in self.pools:
            y.append(pool(x))
        return torch.cat(y, dim=1)


class BaseRNN(nn.Module):
    """
    base class that has memory, each class with hidden state has to derive from BaseRNN
    """

    def __init__(self, hard):
        super(BaseRNN, self).__init__()
        self.sigmoid = self.hard_sigmoid if hard else torch.sigmoid
        self.tanh = self.hard_tanh if hard else torch.tanh
        self.state = {}

    def forward(self, x):
        x = self.x2h(x)
        if x.dim() == 4:
            return self.forward_cell(x)
        else:
            xiseq = x.split(1, 0)
            res = []
            for t, xt in enumerate(xiseq):
                y = self.cell(xt[0])
                res.append(y.unsqueeze(0))
            return torch.cat(res, dim=0)

    def hard_sigmoid(self, x_in):
        x = x_in * 0.5 + 0.5
        y = torch.clamp(x, 0.0, 1.0)
        return y

    def hard_tanh(self, x):
        y = torch.clamp(x, -1.0, 1.0)
        return y

    def x2h(self, x):
        raise NotImplementedError()

    def cell(self, x):
        raise NotImplementedError()

    def get_hidden(self):
        raise NotImplementedError()

    def set_hidden(self):
        raise NotImplementedError()

    def reset(self, mask=None):
        raise NotImplementedError()


class ConvLSTM(BaseRNN):
    r"""ConvLSTM module
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            conv_func=nn.Conv2d,
            hard=False):
        super(ConvLSTM, self).__init__(hard)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = out_channels

        self.conv_x2h = SequenceWise(ConvLayer(in_channels, 4 * out_channels,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               dilation=dilation,
                                               padding=kernel_size // 2,
                                               activation='Identity'))

        self.conv_h2h = conv_func(in_channels=self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=3,
                                  padding=1,
                                  bias=False)

        self.reset()

    def x2h(self, x):
        return self.conv_x2h(x)

    def cell(self, xt):
        if self.prev_h is not None:
            tmp = self.conv_h2h(self.prev_h) + xt
        else:
            tmp = xt
        cc_i, cc_f, cc_o, cc_g = torch.split(tmp, self.hidden_dim, dim=1)
        i = self.sigmoid(cc_i)
        f = self.sigmoid(cc_f)
        o = self.sigmoid(cc_o)
        g = self.tanh(cc_g)
        if self.prev_c is None:
            c = i * g
        else:
            c = f * self.prev_c + i * g
        h = o * self.tanh(c)
        self.prev_h = h
        self.prev_c = c
        return h

    def reset(self, mask=None):
        """To be called in between batches"""
        if mask is None or self.prev_h is None:
            self.prev_h, self.prev_c = None, None
        else:
            self.prev_h, self.prev_c = self.prev_h.detach(), self.prev_c.detach()
            self.prev_h *= mask.to(self.prev_h)
            self.prev_c *= mask.to(self.prev_h)

    def set_hidden(self, prev_h, prev_c):
        self.prev_h = prev_h
        self.prev_c = prev_c

    def get_hidden(self):
        return self.prev_h, self.prev_c


class BottleneckLSTM(BaseRNN):
    # taken from https://github.com/tensorflow/models/blob/master/research/lstm_object_detection/lstm/lstm_cells.py
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, hard=False):
        super(BottleneckLSTM, self).__init__(hard=hard)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = out_channels//2
        self.conv_x2h = SequenceWise(ConvLayer(in_channels, self.hidden_dim, kernel_size, stride, padding, dilation, separable=True))
        self.conv_h2h = SeparableConv2d(self.hidden_dim, self.hidden_dim, kernel_size, 1, padding, dilation)
        self.gates = SeparableConv2d(self.hidden_dim, 4 * self.hidden_dim, 3, 1, 1)
        self.reset()
        self.stride = stride
    
    def x2h(self, x):
        return self.conv_x2h(x)

    def cell(self, xt):   
        bottleneck = xt if self.prev_h is None else xt + self.conv_h2h(self.prev_h)
        bottleneck = self.tanh(bottleneck)
        gates = self.gates(bottleneck)

        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)
        i = self.sigmoid(cc_i)
        f = self.sigmoid(cc_f)
        o = self.sigmoid(cc_o)
        g = self.tanh(cc_g)
        if self.prev_c is None:
            c = i * g
        else:
            c = f * self.prev_c + i * g
        h = o * self.tanh(c)

        output = torch.cat([h, bottleneck], dim=1)
        self.prev_h = h
        self.prev_c = c
        return output
    
    def reset(self, mask=None):
        """To be called in between batches"""
        if mask is None or self.prev_h is None:
            self.prev_h, self.prev_c = None, None
        else:
            self.prev_h, self.prev_c = self.prev_h.detach(), self.prev_c.detach()
            self.prev_h *= mask.to(self.prev_h)
            self.prev_c *= mask.to(self.prev_h)

    def set_hidden(self, prev_h, prev_c):
        self.prev_h = prev_h
        self.prev_c = prev_c

    def get_hidden(self):
        return self.prev_h, self.prev_c


if __name__ == '__main__':
    x = torch.rand(3, 5, 128, 16, 16)

    # sep = SeparableConv2d(128, 128, 3, 1, 1)
    # z = sep(x[0])
    # print(z.shape)

    net = BottleneckLSTM(128, 256)
    y = net(x)
    print(y.shape)