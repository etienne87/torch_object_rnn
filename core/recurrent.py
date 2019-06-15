# pylint: disable-all
import torch.nn as nn
from torch.nn import functional as F
import torch


def time_to_batch(x):
    t, n = x.size()[:2]
    x = x.view(n * t, *x.size()[2:])
    return x, n


def batch_to_time(x, n=32):
    nt = x.size(0)
    time = int(nt / n)
    x = x.view(time, n, *x.size()[1:])
    return x


def hard_sigmoid(x, alpha=0.0):
    return torch.clamp(x + 0.5, 0 - alpha, 1 + alpha)


class GLinear(nn.Module):
    r""" Generic MatMul Operator.
       Applies Distributed in time if you feed a 3D Tensor, it assumes T,N,D format
    """
    def __init__(self, in_channels, out_channels, norm="batch", bias=True, nonlinearity=F.elu):
        super(GLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Linear(in_channels, out_channels, bias=bias)

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
        is_volume = x.dim() == 3
        if is_volume:
            x, n = time_to_batch(x)

        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)

        if is_volume:
            h = batch_to_time(h, n)
        return h


class GConv2d(nn.Module):
    r""" Generic Convolution Operator.
    Applies Distributed in time if you feed a 5D Tensor, it assumes T,N,C,H,W format
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1, separable=True, norm="batch",
                 bias=False, nonlinearity=F.relu):
        super(GConv2d, self).__init__()
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
    return nn.Sequential(GConv2d(in_channels, in_channels, kernel_size, stride, kernel_size // 2, dilation, True, norm),
                         GConv2d(in_channels, out_channels, 1, 1, 1, False, norm))


class BaseRNN(nn.Module):
    """
    base class doing the unrolling, and keeps memory inside
    """

    def __init__(self, hard=False):
        super(BaseRNN, self).__init__()
        self.x2h = lambda x: x
        self.sigmoid = hard_sigmoid if hard else torch.sigmoid
        self.tanh = F.hardtanh if hard else torch.tanh
        self.reset_hidden()

    def forward(self, x):
        xi = self.x2h(x)
        xseq = xi.unbind(0)
        if isinstance(self.prev_hidden, list):
            for item in self.prev_hidden:
                if item is not None:
                    item.detach()
        elif self.prev_hidden is not None:
            self.prev_hidden.detach()

        result = []
        for t, xt in enumerate(xseq):
            self.prev_hidden, hidden = self.update_hidden(xt)
            result.append(hidden)
        result = torch.cat(result, dim=0)
        return result

    def get_hidden(self):
        return self.prev_hidden

    def set_hidden(self, hidden):
        self.prev_hidden = hidden

    def update_hidden(self, xt):
        raise NotImplementedError()

    def reset_hidden(self, mask=None):
        if mask is None or self.prev_hidden is None:
            self.prev_hidden = None
        else:
            if isinstance(self.prev_hidden, list):
                for item in self.prev_hidden:
                    if item is not None:
                        item *= mask
            elif self.prev_hidden is not None:
                self.prev_hidden *= mask


class LSTM(BaseRNN):
    def __init__(self, in_channels, hidden_dim, x2h_func, h2h_func, hard=False, nonlinearity=F.relu):
        super(LSTM, self).__init__(hard)
        self.hidden_dim = hidden_dim
        self.x2h = x2h_func(in_channels, 4 * self.hidden_dim)
        self.h2h = h2h_func(self.hidden_dim, 4 * self.hidden_dim)
        self.act = nonlinearity

    def update_hidden(self, xt):
        prev_h, prev_c = self.prev_hidden if self.prev_hidden is None else None, None
        tmp = xt if prev_h is None else self.h2h(prev_h) + xt
        cc_i, cc_f, cc_o, cc_g = torch.split(tmp, self.hidden_dim, dim=1)
        f = self.sigmoid(cc_f)
        i = self.sigmoid(cc_i)
        o = self.sigmoid(cc_o)
        g = self.act(cc_g)
        c = i * g if prev_h is None else f * prev_c + i * g
        h = o * self.act(c)
        return [h, c], h.unsqueeze(0)


if __name__ == '__main__':
    from functools import partial

    t, n, c, h, w = 4, 4, 8, 64, 64


    # Use-Case: Convolutional RNN
    x = torch.rand(t, n, c, h, w)
    conv_func = partial(GConv2d, kernel_size=5, padding=5//2, norm='weight')
    x2h_func = partial(conv_dw, kernel_size=3, norm='weight', stride=2)
    h2h_func = partial(nn.Conv2d, kernel_size=3, padding=1, stride=1)


    # Use-Case: RNN
    x = torch.rand(t, n, c)
    x2h_func = partial(GLinear, norm='weight', bias=True)
    h2h_func = partial(nn.Linear, bias=True)



    lstm = LSTM(c, 8, x2h_func, h2h_func)
    for _ in range(10):
        lstm.reset_hidden()
        y = lstm(x)
        print(y.shape)

    # conv = conv_func(in_channels=c, out_channels=8, stride=2)

    # print(conv(x).shape)