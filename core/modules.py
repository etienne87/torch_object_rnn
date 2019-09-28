from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from torch.nn import functional as F
import torch
from core.utils.opts import time_to_batch, batch_to_time
from functools import partial

try:
    from inplace_abn import InPlaceABN
except:
    print('ibn not imported, module will use more memory')


def get_padding(kernel, dilation):
    k2 = kernel + (kernel-1) * (dilation-1)
    return k2//2


class DilateSharedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(DilateSharedConv2d, self).__init__(*args, **kwargs)

    def forward(self, sources, rates):
        outputs = []
        for src, rate in zip(sources, rates):
            pad = get_padding(self.kernel_size[0], rate)
            y = F.conv2d(src, self.weight, self.bias, self.stride, pad, rate, self.groups)
            outputs.append(y)
        return outputs



if hasattr(__file__, 'InPlaceABN'):
    class ConvBN(nn.Sequential):

        def __init__(self, in_channels, out_channels,
                     kernel_size=3, stride=1, padding=1, dilation=1,
                     bias=True, activation='leaky_relu'):
            activation = 'identity' if activation=='Identity' else 'identity'
            super(ConvBN, self).__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=padding,
                          bias=bias),
                InPlaceABN(out_channels, activation=activation)
            )
else:
    class ConvBN(nn.Sequential):

        def __init__(self, in_channels, out_channels,
                     kernel_size=3, stride=1, padding=1, dilation=1,
                     bias=True, activation='LeakyReLU'):
            super(ConvBN, self).__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=padding,
                          bias=bias),
                nn.BatchNorm2d(out_channels),
                getattr(nn, activation)()
            )



class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        mid_planes = planes//4
        self.conv1 = ConvBN(in_channels=in_planes, out_channels=mid_planes, kernel_size=1, padding=0, bias=False)
        self.conv2 = ConvBN(in_channels=mid_planes, out_channels=mid_planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.conv3 = ConvBN(in_channels=mid_planes, out_channels=planes, kernel_size=1, padding=0, bias=False)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.downsample = ConvBN(in_channels=in_planes, out_channels=planes,
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
            modules.append(ConvBN(in_channels, out_channels, dilation=rate, padding=rate))

        self.convs = nn.ModuleList(modules)
        self.project = ConvBN(out_channels * len(self.convs), out_channels)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res = self.project(res)
        return res


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

    def __init__(self, hidden_dim, kernel_size, conv_func=nn.Conv2d, hard=False):
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
    def __init__(self, hidden_dim, kernel_size, bias, conv_func=nn.Conv2d, hard=False):
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
    def __init__(self, in_channels, out_channels,
                 kernel_size=5, stride=1, padding=2, dilation=1,
                 cell='gru', hard=False):
        super(ConvRNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if cell == 'gru':
            self.timepool = ConvGRUCell(out_channels, 3, True, hard=hard)
            factor = 3
        else:
            self.timepool = ConvLSTMCell(out_channels, 3, hard=hard)
            factor = 4


        self.conv_x2h = SequenceWise(ConvBN(in_channels, factor * out_channels,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             dilation=dilation,
                                             padding=padding,
                                             activation='Identity'))

    def forward(self, x):
        y = self.conv_x2h(x)
        h = self.timepool(y)
        return h

    def reset(self):
        self.timepool.reset()


class BottleneckRNN(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckRNN, self).__init__()
        mid_planes = planes//4
        self.conv1 = SequenceWise(
            ConvBN(in_channels=in_planes, out_channels=mid_planes, kernel_size=1, padding=0, bias=False))

        self.conv2 = ConvRNN(in_channels=mid_planes, out_channels=mid_planes, kernel_size=3, stride=stride, padding=1)
        self.conv3 = SequenceWise(ConvBN(in_channels=mid_planes, out_channels=planes, kernel_size=1, padding=0, bias=False))

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.downsample = SequenceWise(ConvBN(in_channels=in_planes, out_channels=planes,
                                     kernel_size=1, padding=0, stride=stride,
                          bias=False, activation='Identity'))


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.downsample(x)
        out = F.relu(out)
        return out

    def reset(self):
        self.conv2.reset()


class UpFuse(nn.Module):
    r"""
    Useful for UNet, upscale x1 and merge to x2 by concat or sum
    """
    def __init__(self, scale_factor=2, mode='sum'):
        super(UpFuse, self).__init__()
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, x1, x2):
        x1, n = time_to_batch(x1)
        x1 = F.interpolate(x1, scale_factor=self.scale_factor, mode='nearest')
        x1 = batch_to_time(x1, n)
        if self.mode == 'cat':
            return torch.cat([x1, x2], dim=1)
        else:
            return x1 + x2


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, channels_per_layer,
                        mode='sum', scale_factor=2):
        super(UNet, self).__init__()

        base = channels_per_layer[0]
        self.inc = SequenceWise(ConvBN(in_channels, base))

        self.downs = []
        self.ups = []

        self.mode = mode
        self.channels = [channels_per_layer[0]]
        self.upfuse = UpFuse(scale_factor=scale_factor, mode=mode)

        n_layers = len(channels_per_layer)

        down = partial(ConvRNN, kernel_size=3, stride=2, padding=1, dilation=1)
        up  = partial(ConvRNN, kernel_size=3, stride=1, padding=1, dilation=1)

        # down = partial(BottleneckRNN, stride=2)
        # up = partial(BottleneckRNN, stride=1)


        for i in range(n_layers):
            channels = channels_per_layer[i]
            self.channels.append(channels)
            self.downs.append(down(self.channels[-2], self.channels[-1]))

        for i in range(n_layers):
            in_ch, out_ch = self.channels[-i-1], self.channels[-i-2]
            self.ups.append( up(in_ch, out_ch) )

        self.outc = SequenceWise(nn.Conv2d(out_ch, out_channels, 1))

        self.downs = nn.ModuleList(self.downs)
        self.ups = nn.ModuleList(self.ups)

    def forward(self, x):
        encoded = [self.inc(x)]

        for down_layer in self.downs:
            encoded.append(down_layer(encoded[-1]))

        x = encoded.pop()

        self.decoded = []
        for up_layer in self.ups:
            x = up_layer(x)
            x = self.upfuse(x, encoded.pop())
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

    def __repr__(self):
        repr = ''
        for i, module in enumerate(self.downs):
            repr += 'down#'+str(i)+': '+str(module.in_channels)+";"+str(module.out_channels)+'\n'
        for i, module in enumerate(self.ups):
            repr += 'up#'+str(i)+': '+str(module.in_channels)+";"+str(module.out_channels)+'\n'
        return repr




class FPN(nn.Module):
    def __init__(self, cin=1, cout=128, nmaps=3):
        super(FPN, self).__init__()
        self.cin = cin
        self.base = 8
        self.cout = cout
        self.nmaps = nmaps

        self.conv1 = SequenceWise(nn.Sequential(
            Bottleneck(cin, self.base, 2),
            Bottleneck(self.base, self.base * 8, 2),
            Bottleneck(self.base * 8, self.base * 8, 2)
        ))

        self.conv2 = UNet(self.base * 8,
                               self.cout,
                               channels_per_layer=[self.base * 16, self.base * 16, self.base * 16])

    def forward(self, x):
        x1 = self.conv1(x)
        self.conv2(x1)

        sources = [time_to_batch(item)[0] for item in self.conv2.decoded][::-1]

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
                ConvBN(cin, base, kernel_size=7, stride=2, padding=3),
                ConvBN(base, base * 4, kernel_size=7, stride=2, padding=3),
                ConvBN(base * 4, base * 8, kernel_size=7, stride=2, padding=3)
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







if __name__ == '__main__':
    t, n, c, h, w = 10, 3, 3, 128, 128
    x = torch.rand(t, n, c, h, w)
    # net = UNet(c, 128, channels_per_layer=[128, 128, 128])
    # out = net(x)

    net = FPN(cin=c, cout=128, nmaps=3)
    out = net(x)

    # im = torch.rand(n, c, h, w)
    # net = CosineConv2d(c, 16, kernel_size=3, stride=1, padding=1)
    # y = net.forward(im)
    # print(y.mean(), y.std())
