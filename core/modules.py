from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from torch.nn import functional as F
import torch
from core.utils.opts import time_to_batch, batch_to_time
from functools import partial



class ConvBN(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, dilation=1,
                 bias=True, act=nn.ReLU6(inplace=True)):
        super(ConvBN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=padding,
                               bias=bias)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = act

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        return x


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        If RNN (has reset module), will bypass reshapes
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = time_to_batch(x)[0]
        x = self.module(x)
        x = batch_to_time(x, n)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class RNNCell(nn.Module):
    """
    base class that has memory, each class with hidden state has to derive from basernn
    """

    def __init__(self, hard):
        super(RNNCell, self).__init__()
        self.saturation_cost = 0
        self.saturation_limit = 0.9
        self.saturation_weight = 1e-4
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
        cost = sat_loss.sum()
        cost *= self.saturation_weight
        self.saturation_cost += cost

    def reset(self):
        raise NotImplementedError()


class ConvLSTMCell(RNNCell):
    r"""ConvLSTMCell module, applies sequential part of LSTM.
    """

    def __init__(self, hidden_dim, kernel_size, conv_func=ConvBN, hard=False):
        super(ConvLSTMCell, self).__init__(hard)
        self.hidden_dim = hidden_dim

        self.conv_h2h = conv_func(in_channels=self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=kernel_size,
                                  padding=1,
                                  bias=False)

        self.reset()

    def forward(self, xi):
        self.saturation_cost = 0
        inference = (len(xi.shape) == 4)  # inference for model conversion
        if inference:
            xiseq = [xi]
        else:
            xiseq = xi.split(1, 0)  # t,n,c,h,w

        if self.prev_h is not None:
            self.prev_h = self.prev_h.detach()
            self.prev_c = self.prev_c.detach()

        result = []
        for t, xt in enumerate(xiseq):
            if not inference:  # for training/val (not inference)
                xt = xt.squeeze(0)

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
            if not inference:
                result.append(h.unsqueeze(0))
            self.prev_h = h
            self.prev_c = c
        if inference:
            res = h
        else:
            res = torch.cat(result, dim=0)
        return res

    def reset(self):
        self.prev_h, self.prev_c = None, None


class ConvGRUCell(RNNCell):
    r"""ConvGRUCell module, applies sequential part of LSTM.
    """
    def __init__(self, hidden_dim, kernel_size, bias, conv_func=ConvBN, hard=False):
        super(ConvGRUCell, self).__init__(hard)
        self.hidden_dim = hidden_dim

        # Fully-Gated
        self.conv_h2zr = conv_func(in_channels=self.hidden_dim,
                                  out_channels=2 * self.hidden_dim,
                                  kernel_size=kernel_size,
                                  padding=1,
                                  bias=bias)

        self.conv_h2h = conv_func(in_channels=self.hidden_dim,
                                  out_channels=self.hidden_dim,
                                  kernel_size=kernel_size,
                                  padding=1,
                                  bias=bias)

        self.reset()

    def forward(self, xi):
        self.saturation_cost = 0

        xiseq = xi.split(1, 0) #t,n,c,h,w


        if self.prev_h is not None:
            self.prev_h = self.prev_h.detach()

        result = []
        for t, xt in enumerate(xiseq):
            xt = xt.squeeze(0)


            #split x & h in 3
            x_zr, x_h = xt[:, :2*self.hidden_dim], xt[:,2*self.hidden_dim:]

            if self.prev_h is not None:
                tmp = self.conv_h2zr(self.prev_h) + x_zr
            else:
                tmp = x_zr

            cc_z, cc_r = torch.split(tmp, self.hidden_dim, dim=1)
            z = self.sigmoid(cc_z)
            r = self.sigmoid(cc_r)

            if self.prev_h is not None:
                tmp = self.conv_h2h(r * self.prev_h) + x_h
            else:
                tmp = x_h
            tmp = self.tanh(tmp)

            if self.prev_h is not None:
                h = (1-z) * self.prev_h + z * tmp
            else:
                h = z * tmp


            result.append(h.unsqueeze(0))
            self.prev_h = h
        res = torch.cat(result, dim=0)
        return res

    def reset(self):
        self.prev_h = None


class ConvRNN(nn.Module):
    r"""ConvRNN module.
    """
    def __init__(self, nInputPlane, nOutputPlane,
                 kernel_size=5, stride=1, padding=2, dilation=1,
                 cell='gru', hard=True):
        super(ConvRNN, self).__init__()

        self.cin = nInputPlane
        self.cout = nOutputPlane
        if cell == 'gru':
            self.timepool = ConvGRUCell(nOutputPlane, 3, True, hard=hard)
            factor = 3
        else:
            self.timepool = ConvLSTMCell(nOutputPlane, 3, True, hard=hard)
            factor = 4

        self.conv_x2h = SequenceWise(ConvBN(nInputPlane, factor * nOutputPlane,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            dilation=dilation,
                                            padding=padding,
                                            act=nn.Identity()))

    def forward(self, x):
        y = self.conv_x2h(x)
        h = self.timepool(y)
        return h

    def reset(self):
        self.timepool.reset()


class UpFuseRNN(nn.Module):
    def __init__(self, in_ch, out_ch, mode='cat'):
        super(UpFuseRNN, self).__init__()
        self.mode = mode
        self.convrnn = ConvRNN(in_ch, out_ch, 3, 1, 1)

    def forward(self, x1, x2):
        h, w = x2.shape[-2:]
        x1, n = time_to_batch(x1)
        x1 = F.interpolate(x1, size=(h, w), mode='bilinear', align_corners=True)
        x1 = batch_to_time(x1, n)
        x = x1 + x2
        return self.convrnn(x)

    def reset(self):
        self.convrnn.reset()


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base, n_layers):
        super(UNet, self).__init__()

        self.inc = SequenceWise(ConvBN(in_channels, base))

        self.downs = []
        self.ups = []

        self.channels = [base]

        down = partial(ConvRNN, kernel_size=3, stride=2, padding=1, dilation=1)
        up  = UpFuseRNN

        for i in range(n_layers):
            channels = min(base * 2 ** (n_layers-1), self.channels[-1] * 2)
            self.channels.append(channels)
            self.downs.append( down(self.channels[-2], self.channels[-1]) )

        self.channels.pop()

        for i in range(n_layers):
            channels = self.channels.pop()
            in_ch = channels
            out_ch = max(base, channels // 2)
            self.ups.append( up(in_ch, out_ch) )

        self.outc = SequenceWise(nn.Conv2d(base, out_channels, 1))

        self.downs = nn.ModuleList(self.downs)
        self.ups = nn.ModuleList(self.ups)


    def forward(self, x):
        encoded = [self.inc(x)]

        for down_layer in self.downs:
            encoded.append(down_layer(encoded[-1]))

        x = encoded.pop()

        self.decoded = []
        for up_layer in self.ups:
            x = up_layer(x, encoded.pop())
            self.decoded.append(x)

        x = self.outc(x)
        self.decoded.append(x)

        return x

    def reset(self):
        for module in self.downs:
            if hasattr(module, "reset"):
                module.reset()

        for module in self.ups:
            if hasattr(module, "reset"):
                module.reset()


if __name__ == '__main__':
    t, n, c, h, w = 10, 1, 3, 64, 64

    x = torch.rand(t, n, c, h, w)

    net = nn.Sequential(
            UNet(c, 64, 16, 3),
            UNet(64, 64, 16, 3))

    out = net(x)

    net[0].reset()
    net[1].reset()
