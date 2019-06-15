import torch
import torch.nn as nn
from torch.nn import functional as F
import recurrent as rnn
from functools import partial

def time_to_batch(x):
    t, n = x.size()[:2]
    x = x.view(n * t, *x.size()[2:])
    return x, n


def batch_to_time(x, n=32):
    nt = x.size(0)
    time = int(nt / n)
    x = x.view(time, n, *x.size()[1:])
    return x

class Conv2d(nn.Module):
    r""" Generic Convolution Operator.
    Applies Distributed in time if you feed a 5D Tensor, it assumes T,N,C,H,W format
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1, separable=False, norm="none",
                 bias=False, nonlinearity=lambda x:x):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               groups=in_channels if separable else 1,
                               padding=(kernel_size * dilation) // 2,
                               bias=bias)

        if norm == "batch":
            self.bn1 = nn.BatchNorm2d(in_channels, affine=True)
        elif norm == "instance":
            self.bn1 = nn.InstanceNorm2d(in_channels, affine=True)
        elif norm == "group":
            self.bn1 = nn.GroupNorm(num_groups=3, num_channels=in_channels, affine=True)
        elif norm == "weight":
            self.conv1 = nn.utils.weight_norm(self.conv1)
            self.bn1 = lambda x: x
        else:
            self.bn1 = lambda x: x

        self.act = nonlinearity

    def forward(self, x):
        is_volume = x.dim() == 5
        if is_volume:
            x, n = time_to_batch(x)

        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)

        if is_volume:
            h = batch_to_time(h, n)
        return h


def conv_dw(in_channels, out_channels, kernel_size=3, stride=1, norm='instance', dilation=1):
    return nn.Sequential(Conv2d(in_channels, in_channels, kernel_size, stride, kernel_size // 2, dilation, True, norm),
                         Conv2d(in_channels, out_channels, 1, 1, 1, False, norm))


def lstm_conv(cin, cout, stride=1, dilation=1):
    oph = partial(Conv2d, kernel_size=3, stride=1, bias=True, dilation=dilation)
    opx = partial(conv_dw, kernel_size=3, norm='batch', stride=stride, dilation=dilation)
    return rnn.LSTMCell(cin, cout, opx, oph, nonlinearity=F.relu)



class TridentRNN(nn.Module):
    def __init__(self, cin=1):
        super(TridentRNN, self).__init__()
        self.cin = cin
        base = 8

        self.conv1 = Conv2d(cin, base, kernel_size=7, stride=2)
        self.conv2 = conv_dw(base, base * 2, kernel_size=7, stride=2)

        self.conv3 = lstm_conv(base * 2, base * 4, dilation=1)
        self.conv4 = lstm_conv(base * 4, base * 4, dilation=2)
        self.conv5 = lstm_conv(base * 4, base * 4, dilation=4)
        self.conv6 = lstm_conv(base * 4, base * 4, dilation=8)

        self.end_point_channels = [self.conv3.cout,  # 8
                                   self.conv4.cout,  # 16
                                   self.conv5.cout,  # 32
                                   self.conv6.cout]  # 64

    def forward(self, x):
        sources = list()
        x = self.conv1(x)
        x = self.conv2(x)

        x3 = self.conv3(x)
        x4 = self.conv4(x3)
        x5 = self.conv5(x)
        x6 = self.conv6(x)

        x3, n = time_to_batch(x3)
        x4, n = time_to_batch(x4)
        x5, n = time_to_batch(x5)
        x6, n = time_to_batch(x6)

        sources += [x3, x4, x5, x6]
        return sources

    def reset(self, mask=None):
        for name, module in self._modules.iteritems():
            if isinstance(module, RNN):
                module.reset(mask)

if __name__ == '__main__':
