import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch


def time_to_batch(x):
    n, c, t, h, w = x.size()
    x = x.permute([0, 2, 1, 3, 4]).contiguous().view(n * t, c, h, w)
    return x, n


def batch_to_time(x, n=32):
    nt, c, h, w = x.size()
    t = int(nt / n)
    x = x.view(n, t, c, h, w).permute([0, 2, 1, 3, 4]).contiguous()
    return x

class ConvLSTMCell(nn.Module):
    r"""ConvLSTMCell module, applies sequential part of LSTM.
    """
    def __init__(self, hidden_dim, kernel_size, bias, nonlin=F.leaky_relu):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim

        self.conv_h2h = nn.Conv2d(in_channels=self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=kernel_size,
                                  padding=1,
                                  bias=bias)

        self.reset()
        self.nonlin = nonlin

    def forward(self, xi):
        xiseq = xi.split(1, 2)  # n,c,t,h,w

        if self.prev_h is not None:
            self.prev_h = self.prev_h.detach()
            self.prev_c = self.prev_c.detach()

        result = []
        for t, xt in enumerate(xiseq):
            if self.prev_h is None:
                tmp = xt.squeeze(2)
            else:
                tmp = self.conv_h2h(self.prev_h) + xt.squeeze()
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
            result.append(h.unsqueeze(2))
            self.prev_h = h
            self.prev_c = c
        res = torch.cat(result, dim=2)
        return res

    def reset(self):
        self.prev_h, self.prev_c = None, None


class ConvLSTM(nn.Module):
    r"""ConvLSTM module. computes input-to-hidden in parallel.
    """
    def __init__(self, nInputPlane, nOutputPlane, kernel_size, stride, padding, dilation=1):
        super(ConvLSTM, self).__init__()

        self.cin = nInputPlane
        self.cout = nOutputPlane
        self.conv1 = nn.Conv2d(nInputPlane, 4 * nOutputPlane, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=padding,
                               bias=True)

        self.bn1 = nn.BatchNorm2d(nInputPlane)

        self.timepool = ConvLSTMCell(nOutputPlane, 3, True)

    def forward(self, x):
        x, n = time_to_batch(x)
        bnx = self.bn1(x)
        y = self.conv1(bnx)
        y = batch_to_time(y, n)
        h = self.timepool(y)
        return h


class ConvLSTM2(torch.nn.Module):

    def __init__(self, cin=3, cout=10, kernel_size=7, stride=1, padding=3):
        super(ConvLSTM2, self).__init__()
        self.cout = cout
        self.bnxh = nn.BatchNorm2d(cin)
        self.convxh = nn.Conv2d(cin, 4 * cout, kernel_size, stride, padding, bias=True)

        self.convhh = nn.Conv2d(cout, 4 * cout, 3, 1, 1, bias=False)
        self.reset()

    def forward(self, input):

        if self.h is not None:
            h, c = self.h.detach(), self.c.detach()
        else:
            h, c = None, None


        res = []
        for t in range(input.size(2)):
            xt = input[:, :, t]
            tmp = self.convxh(self.bnxh(input[:,:,t]))
            if h is not None:
                tmp = tmp + self.convhh(h)
            i = torch.sigmoid(tmp[:, 0:self.cout])
            f = torch.sigmoid(tmp[:, self.cout: self.cout * 2])
            o = torch.sigmoid(tmp[:, self.cout * 2:self.cout * 3])
            g = torch.sigmoid(tmp[:, self.cout * 3:self.cout * 4])
            if h is None:
                c = i * g
            else:
                c = f * c + i * g
            h = o * nn.functional.leaky_relu(c)
            res.append(h)

        self.h, self.c = h, c
        res = torch.cat([item.unsqueeze(2) for item in res], dim=2)
        return res

    def reset(self):
        self.h, self.c = None, None


class ForgetMult(nn.Module):
    r"""ForgetMult computes a simple recurrent equation:
        h_t = f_t * x_t + (1 - f_t) * h_{t-1}
        This equation is equivalent to dynamic weighted averaging.
    """

    def __init__(self):
        super(ForgetMult, self).__init__()
        self.prev_h = None

    def forward(self, xi):
        if self.prev_h is not None:
            self.prev_h = self.prev_h.detach()

        result = []
        f, x = xi.split(xi.size(1) / 2, dim=1)
        f = F.sigmoid(f)
        forgets = f.split(1, dim=2)
        fx = (f * x).split(1, dim=2)
        for i, h in enumerate(fx):
            if self.prev_h is not None: h = h + (1 - forgets[i]) * self.prev_h
            result.append(h)
            self.prev_h = h
        res = torch.cat(result, dim=2)
        return res

    def reset(self):
        self.prev_h = None


class ConvQRNN(nn.Module):
    r"""ConvQRNN module. forget gates are precomputed in parallel.
    """
    def __init__(self, nInputPlane, nOutputPlane, kernel_size, stride, padding, dilation=1):
        super(ConvQRNN, self).__init__()
        self.Cin = nInputPlane
        self.Cout = nOutputPlane
        self.conv1 = nn.Conv3d(nInputPlane, nOutputPlane * 2, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=padding,
                               bias=True)
        self.bn1 = nn.BatchNorm3d(nInputPlane)
        self.timepool = ForgetMult()

    def forward(self, x):
        bnx = self.bn1(x)
        y = self.conv1(bnx)
        h = self.timepool(y)
        h = F.relu(h)
        return h

    def reset(self):
        self.timepool.reset()
