from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from torch.nn import functional as F
import torch
from utils.opts import time_to_batch, batch_to_time



class Conv2d(nn.Module):

    def __init__(self, cin, cout, kernel_size, stride, padding, dilation=1, act=nn.ReLU(inplace=True)):
        super(Conv2d, self).__init__()
        self.cin = cin
        self.cout = cout
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=padding,
                               bias=True)

        self.bn1 = nn.BatchNorm2d(cout)
        self.act = act

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        return x


class ResNetBlock(nn.Module):
    """Transition block"""

    def __init__(self, cin):
        super(ResNetBlock, self).__init__()
        self.conv1 = Conv2d(cin, cin, 3, padding=1, stride=1)
        self.conv2 = Conv2d(cin, cin, 3, padding=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        return self.conv2(conv1) + x


def double_conv(in_ch, out_ch):
    return nn.Sequential(
            Conv2d(in_ch, out_ch, 3, padding=1, stride=1),
            Conv2d(out_ch, out_ch, 3, padding=1, stride=1)
    )


def down_ff(in_ch, out_ch):
    return nn.Sequential(
        nn.MaxPool2d(2),
        double_conv(in_ch, out_ch)
    )


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
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


class ConvLSTMCell(nn.Module):
    r"""ConvLSTMCell module, applies sequential part of LSTM.
    """

    def __init__(self, hidden_dim, kernel_size, conv_func=nn.Conv2d, nonlin=torch.tanh):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim

        self.conv_h2h = conv_func(in_channels=self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=kernel_size,
                                  padding=1,
                                  bias=False)

        self.reset()
        self.nonlin = nonlin

    def forward(self, xi):
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
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            if self.prev_c is None:
                c = i * g
            else:
                c = f * self.prev_c + i * g
            h = o * self.nonlin(c)
            if not inference:
                result.append(h.unsqueeze(0))
            self.prev_h = h
            self.prev_c = c
        if inference:
            res = h
        else:
            res = torch.cat(result, dim=0)
        return res

    def set(self, prev_h, prev_c):
        self.prev_h = prev_h
        self.prev_c = prev_c

    def get(self):
        return self.prev_h, self.prev_c

    def reset(self):
        self.prev_h, self.prev_c = None, None


class ConvLSTM(nn.Module):
    r"""ConvLSTM module.
    """
    def __init__(self, nInputPlane, nOutputPlane, kernel_size, stride, padding, dilation=1):
        super(ConvLSTM, self).__init__()

        self.cin = nInputPlane
        self.cout = nOutputPlane

        self.conv_x2h = SequenceWise(Conv2d(nInputPlane, 4 * nOutputPlane,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            dilation=dilation,
                                           padding=padding,
                                           act=nn.Identity()))

        self.timepool = ConvLSTMCell(nOutputPlane, 3)

    def forward(self, x):
        y = self.conv_x2h(x)
        h = self.timepool(y)
        return h



class ConvGRUCell(nn.Module):
    r"""ConvGRUCell module, applies sequential part of LSTM.
    """
    def __init__(self, hidden_dim, kernel_size, bias):
        super(ConvGRUCell, self).__init__()
        self.hidden_dim = hidden_dim

        # Fully-Gated
        self.conv_h2zr = nn.Conv2d(in_channels=self.hidden_dim,
                                  out_channels=2 * self.hidden_dim,
                                  kernel_size=kernel_size,
                                  padding=1,
                                  bias=bias)

        self.conv_h2h = nn.Conv2d(in_channels=self.hidden_dim,
                                  out_channels=self.hidden_dim,
                                  kernel_size=kernel_size,
                                  padding=1,
                                  bias=bias)

        self.reset()


    def forward(self, xi):
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
            z = torch.sigmoid(cc_z)
            r = torch.sigmoid(cc_r)

            if self.prev_h is not None:
                tmp = self.conv_h2h(r * self.prev_h) + x_h
            else:
                tmp = x_h
            tmp = torch.tanh(tmp)

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


class ConvGRU(nn.Module):
    r"""ConvGRU module.
    """
    def __init__(self, nInputPlane, nOutputPlane, kernel_size=5, stride=1, padding=2, dilation=1):
        super(ConvGRU, self).__init__()

        self.cin = nInputPlane
        self.cout = nOutputPlane
        self.conv_x2h = SequenceWise(Conv2d(nInputPlane, 3 * nOutputPlane,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            dilation=dilation,
                                            padding=padding,
                                            act=nn.Identity()))

        self.timepool = ConvGRUCell(nOutputPlane, 3, True)

    def forward(self, x):
        y = self.conv_x2h(x)
        h = self.timepool(y)
        return h


class UpCatRNN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpCatRNN, self).__init__()
        self.convrnn = ConvGRU(in_ch, out_ch)

    def forward(self, x1, x2):

        x1, n = time_to_batch(x1)
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)

        # input is CHW
        diffY = x2.size()[-2] - x1.size()[-2]
        diffX = x2.size()[-1] - x1.size()[-1]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x1 = batch_to_time(x1, n)

        x = torch.cat([x2, x1], dim=2)
        x = self.convrnn(x)
        return x



class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base, n_layers, up, down):
        super(UNet, self).__init__()

        self.inc = SequenceWise(double_conv(in_channels, base))

        self.downs = []
        self.ups = []

        self.channels = [base]

        for i in range(n_layers):
            channels = min(base * 2 ** (n_layers-1), self.channels[-1] * 2)
            self.channels.append(channels)
            self.downs.append( SequenceWise(down(self.channels[-2], self.channels[-1])) )

        self.channels.pop()

        for i in range(n_layers):
            channels = self.channels.pop()
            in_ch = channels * 2
            out_ch = max(base, channels // 2)
            self.ups.append(up(in_ch, out_ch) )

        self.outc = SequenceWise(nn.Conv2d(base, out_channels, 1))

        self.downs = nn.ModuleList(self.downs)
        self.ups = nn.ModuleList(self.ups)


    def forward(self, x):
        self.encoded = [self.inc(x)]

        for down_layer in self.downs:
            self.encoded.append(down_layer(self.encoded[-1]))

        x = self.encoded.pop()

        for up_layer in self.ups:
            x = up_layer(x, self.encoded.pop())

        x = self.outc(x)
        return x




if __name__ == '__main__':
    t, n, c, h, w = 10, 1, 3, 64, 64

    x = torch.rand(t, n, c, h, w)

    net = nn.Sequential(
            UNet(c, 64, 16, 3, up=UpCatRNN, down=down_ff),
            UNet(64, 64, 16, 3, up=UpCatRNN, down=down_ff))

    out = net(x)