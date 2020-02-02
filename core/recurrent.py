from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch
from core.utils.opts import time_to_batch, batch_to_time
from core.modules import ConvLayer




class RNNWise(nn.Sequential):
    """ Calls whole architecture in a RNN fashion
        Every layer is a cell fed with input at time t.
        The architecture can output a tensor or a list of tensors.
        This allows feedback architecture for example.
        
        Warning: This design can be slower than module's "ConvRNN"
        since no sequence-in-parallel computation can occur.
    """
    def __init__(self, *args):
        super(RNNWise, self).__init__(*args)

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
        self.detach_modules()
        xiseq = x.split(1, 0)
        res = []
        for t, xt in enumerate(xiseq):
            y = super().forward(xt[0])
            res.append(y)

        return self.pack_results(res)

    def reset_modules(self, mask=None):
        for module in self.modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    def detach_modules(self):
        for module in self.modules():
            if hasattr(module, "detach"):
                module.detach()


class RNNCell(nn.Module):
    """
    base class that has memory, each class with hidden state has to derive from RNNCell
    TODO: add the saturation cost as a hook!!

    """

    def __init__(self, hard=False):
        super(RNNCell, self).__init__()
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

    def reset(self, mask=None):
        raise Exception("Not Implemented")

    def detach(self):
        raise Exception("Not Implemented")


class ConvLSTMCell(RNNCell):
    r"""ConvLSTMCell module, applies sequential part of LSTM.
    """

    def __init__(self, in_channels, 
                       out_channels, 
                       kernel_size=3, stride=1, padding=1, dilation=1, bias=True, 
                       conv_func=nn.Conv2d, hard=False, feedback_channels=None):
        super(ConvLSTMCell, self).__init__(hard)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_x2h = ConvLayer(in_channels, self.out_channels * 4, 
        activation='Identity', stride=stride)

        self.conv_h2h = conv_func(in_channels=self.out_channels,
                                  out_channels=4 * self.out_channels,
                                  kernel_size=kernel_size,
                                  stride=1,
                                  padding=padding,
                                  dilation=dilation,
                                  bias=bias)

        if feedback_channels is not None:
            self.conv_fb2h = conv_func(in_channels=feedback_channels,
                                  out_channels=4 * self.out_channels,
                                  kernel_size=kernel_size,
                                  stride=1,
                                  padding=padding,
                                  dilation=dilation,
                                  bias=bias)

        self.stride = stride

        self.reset()

    def forward(self, x):
        x = self.conv_x2h(x)

        if self.prev_h is None:
            shape = list(x.shape)
            shape[1] = self.out_channels
            self.prev_h = torch.zeros(shape, dtype=torch.float32, device=x.device)
            self.prev_c = torch.zeros(shape, dtype=torch.float32, device=x.device)

        tmp = self.conv_h2h(self.prev_h) + x

        if self.prev_fb is not None:
            tmp += self.conv_fb2h(self.prev_fb)

        cc_i, cc_f, cc_o, cc_g = torch.split(tmp, self.out_channels, dim=1)
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
        self.saturation_cost = 0
        if self.prev_h is not None:
            self.prev_h = self.prev_h.detach()
            self.prev_c = self.prev_c.detach()
        if self.prev_fb is not None:
            self.prev_fb = self.prev_fb.detach()

    def reset(self, mask=None):
        if mask is None or self.prev_h is None:
            self.prev_h, self.prev_c = None, None
            self.prev_fb = None
        else:
            mask = mask.to(self.prev_h)
            self.prev_h *= mask
            self.prev_c *= mask
            if self.prev_fb is not None:
                self.prev_fb *= mask




if __name__ == '__main__':
    x = torch.rand(3, 5, 128, 8, 8)

    net = RNNWise(ConvLSTMCell(128, 256, stride=2), ConvLSTMCell(256, 256, stride=2))
    net.reset()
    y = net(x)
    print(y.shape)