from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch
from core.utils.opts import time_to_batch, batch_to_time



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

    def pack_results(self, outs):
        if isinstance(outs[0], torch.Tensor):
            outs = [item.unsqueeze(0) for item in outs]
            outs = torch.cat(outs, dim=0)
        elif isinstance(outs[0], list):
            t, n = len(outs), len(outs[0])
            res = [[outs[j][i] for j in range(t)] for i in range(n)]
            outs = [self.pack_results(item) for item in res]
        return outs

    def forward(self, x):
        xiseq = x.split(1, 0)
        res = []
        for t, xt in enumerate(xiseq):
            y = super().forward(xt[0])
            res.append(y)

        return self.pack_results(res)


class RNNWise(SequenceWise):
    def __init__(self, *args):
        super(RNNWise, self).__init__(*args)

    def forward(self, x):
        self.detach_modules()
        return super().forward(x)

    def reset_modules(self):
        for module in self.modules():
            if hasattr(module, "reset"):
                module.reset()

    def detach_modules(self):
        for module in self.modules():
            if hasattr(module, "detach"):
                module.detach()


class RNN(nn.Module):
    """
    base class that has memory, each class with hidden state has to derive from RNN
    TODO: add the saturation cost as a hook!!

    """

    def __init__(self, hard=False):
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
        raise Exception("Not Implemented")

    def detach(self):
        raise Exception("Not Implemented")


class ConvLSTM(RNN):
    r"""ConvLSTMCell module, applies sequential part of LSTM.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, conv_func=nn.Conv2d, hard=False):
        super(ConvLSTM, self).__init__(hard)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_h2h = conv_func(in_channels=in_channels + self.out_channels,
                                  out_channels=4 * self.out_channels,
                                  kernel_size=kernel_size,
                                  stride=1,
                                  padding=padding,
                                  dilation=dilation,
                                  bias=bias)

        self.stride = stride

        self.reset()

    def forward(self, x):
        if self.prev_h is None:
            shape = list(x.shape)
            shape[1] = self.out_channels
            self.prev_h = torch.zeros(shape, dtype=torch.float32, device=x.device)
            self.prev_c = torch.zeros(shape, dtype=torch.float32, device=x.device)

        xh = torch.cat([x,self.prev_h], dim=1)
        tmp = self.conv_h2h(xh)

        cc_i, cc_f, cc_o, cc_g = torch.split(tmp, self.out_channels, dim=1)
        i = self.sigmoid(cc_i)
        f = self.sigmoid(cc_f)
        o = self.sigmoid(cc_o)
        g = self.tanh(cc_g)
        c = f * self.prev_c + i * g
        h = o * self.tanh(c)
        self.prev_c = c
        self.prev_h = h

        if self.stride > 1:
            h = F.interpolate(h, scale_factor=1./self.stride, mode='bilinear', align_corners=True)

        return h

    def detach(self):
        self.saturation_cost = 0
        if self.prev_h is not None:
            self.prev_h = self.prev_h.detach()
            self.prev_c = self.prev_c.detach()

    def reset(self):
        self.prev_h, self.prev_c = None, None




if __name__ == '__main__':
    x = torch.rand(3, 5, 128, 8, 8)

    net = RNNWise(ConvLSTM(128, 256, stride=2), ConvLSTM(256, 256, stride=2))

    net.reset_modules()
    #net = ParallelWise(nn.Conv2d(128, 128, 1, 1, 0), nn.ReLU())


    y = net(x)
    print(y.shape)