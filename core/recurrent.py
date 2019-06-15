# pylint: disable-all
import torch.nn as nn
from torch.nn import functional as F
import torch
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


def hard_sigmoid(x, alpha=0.0):
    return torch.clamp(x + 0.5, 0 - alpha, 1 + alpha)


class Linear(nn.Module):
    r""" Generic MatMul Operator.
       Applies Distributed in time if you feed a 3D Tensor, it assumes T,N,D format
    """
    def __init__(self, in_channels, out_channels, norm="none", bias=True, nonlinearity=lambda x:x):
        super(Linear, self).__init__()
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


class Conv2d(nn.Module):
    r""" Generic Convolution Operator.
    Applies Distributed in time if you feed a 5D Tensor, it assumes T,N,C,H,W format
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1, separable=True, norm="none",
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


class RNN(nn.Module):
    """
    base class doing the unrolling
    """

    def __init__(self, cell, hard=False):
        super(RNN, self).__init__()
        # user can define a parallel input-to-hidden operation
        self.sigmoid = hard_sigmoid if hard else torch.sigmoid
        self.tanh = F.hardtanh if hard else torch.tanh
        self.cell = cell
        self.reset()

    def forward(self, x, future=0):
        xseq = x.unbind(0)
        # self.detach_hidden()
        for item in self.cell:
            if isinstance(item, RNNCell):
                item.prev_hidden = None

        result = []
        #First treat sequence
        for t, xt in enumerate(xseq):
            ht = self.cell(xt)
            result.append(ht[None])

        #For auto-regressive use-cases
        if future:
            assert ht.shape[1] == xt.shape[1]
            for _ in range(future):
                ht = self.cell(ht)
                result.append(ht[None])

        result = torch.cat(result, dim=0)
        return result

    def detach_hidden(self):
        for name, module in self._modules.iteritems():
            if isinstance(module, RNNCell):
                module.detach_hidden()

    def reset(self, mask=None):
        for name, module in self._modules.iteritems():
            if isinstance(module, RNNCell):
                module.reset(mask)



class RNNCell(nn.Module):
    r"""
    defines minimum behavior of recurrent cell
    """
    def __init__(self, hard):
        super(RNNCell, self).__init__()
        self.sigmoid = hard_sigmoid if hard else torch.sigmoid
        self.tanh = F.hardtanh if hard else torch.tanh

    def detach_hidden(self):
        #ERROR HERE!!!
        if isinstance(self.prev_hidden, list):
            for item in self.prev_hidden:
                if item is not None:
                    item.detach()
        elif self.prev_hidden is not None:
            self.prev_hidden.detach()

    def reset(self, mask=None):
        if mask is None or self.prev_hidden is None:
            self.prev_hidden = None
        else:
            if isinstance(self.prev_hidden, list):
                for item in self.prev_hidden:
                    if item is not None:
                        item *= mask
            elif self.prev_hidden is not None:
                self.prev_hidden *= mask


class LSTMCell(RNNCell):
    def __init__(self, in_channels, hidden_dim, x2h_func, h2h_func, hard=False, nonlinearity=F.relu):
        super(LSTMCell, self).__init__(hard)
        self.hidden_dim = hidden_dim
        self.x2h = x2h_func(in_channels, 4 * self.hidden_dim)
        self.h2h = h2h_func(self.hidden_dim, 4 * self.hidden_dim)
        self.act = nonlinearity
        self.prev_hidden = None
        self.reset()

    def forward(self, xt):
        if self.prev_hidden is None:
            prev_h = torch.zeros(xt.size(0), self.hidden_dim, dtype=torch.double, device=xt.device)
            prev_c = torch.zeros(xt.size(0), self.hidden_dim, dtype=torch.double, device=xt.device)
        else:
            prev_h, prev_c = self.prev_hidden

        tmp = self.x2h(xt) + self.h2h(prev_h)
        cc_i, cc_f, cc_o, cc_g = torch.split(tmp, self.hidden_dim, dim=1)
        f = torch.sigmoid(cc_f)
        i = torch.sigmoid(cc_i)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c = f * prev_c + i * g
        h = o * torch.tanh(c)

        self.prev_hidden = [h, c]

        return h

        # xt = self.x2h(xt)
        # prev_h, prev_c = self.prev_hidden if self.prev_hidden is None else None, None
        # tmp = xt if prev_h is None else self.h2h(prev_h) + xt
        # cc_i, cc_f, cc_o, cc_g = torch.split(tmp, self.hidden_dim, dim=1)
        # f = torch.sigmoid(cc_f)
        # i = torch.sigmoid(cc_i)
        # o = torch.sigmoid(cc_o)
        # g = self.act(cc_g)
        # c = i * g if prev_h is None else f * prev_c + i * g
        # h = o * self.act(c)
        # self.prev_hidden = [h, c]
        # return h


class OfficialLSTMCell(RNNCell):
    def __init__(self, in_channels, hidden_dim):
        super(OfficialLSTMCell, self).__init__(False)
        self.hidden_dim = hidden_dim
        self.op = nn.LSTMCell(in_channels, hidden_dim)
        self.reset()

    def forward(self, xt):
        if self.prev_hidden is None:
            prev_h = torch.zeros(xt.size(0), self.hidden_dim, dtype=torch.double, device=xt.device)
            prev_c = torch.zeros(xt.size(0), self.hidden_dim, dtype=torch.double, device=xt.device)
        else:
            prev_h, prev_c = self.prev_hidden

        ht, ct = self.op(xt, (prev_h, prev_c))

        self.prev_hidden = [ht, ct]

        return ht



def lstm_fc(cin, cout):
    #return OfficialLSTMCell(cin, cout)
    op = partial(nn.Linear, bias=True)
    return LSTMCell(cin, cout, op, op, nonlinearity=torch.tanh)


if __name__ == '__main__':
    import time

    t, n, c, h, w = 100, 4, 8, 64, 64
    cuda = True

    # Use-Case: Convolutional RNN
    x = torch.rand(t, n, c, h, w)
    conv_func = partial(Conv2d, kernel_size=5, padding=5//2, norm='weight')
    x2h_func = partial(conv_dw, kernel_size=3, norm='weight', stride=1)
    h2h_func = partial(nn.Conv2d, kernel_size=3, padding=1, stride=1)


    # Use-Case: RNN
    # x = torch.rand(t, n, c)
    # x2h_func = partial(Linear, norm='weight', bias=True)
    # h2h_func = partial(nn.Linear, bias=True)

    # lstm = RNN(c, 8, x2h_func, h2h_func)
    #
    #
    # if cuda:
    #     x = x.cuda()
    #     lstm = lstm.cuda()
    #
    # start = time.time()
    # for _ in range(10):
    #     lstm.reset_hidden()
    #     y = lstm(x)
    #
    # print(time.time()-start)

    # conv = conv_func(in_channels=c, out_channels=8, stride=2)

    # print(conv(x).shape)