# pylint: disable-all
import torch.nn as nn
from torch.nn import functional as F
import torch
from functools import partial


def time_to_batch(x):
    t, n = x.size()[:2]
    x = x.view(n * t, *x.size()[2:])
    return x

def batch_to_time(x, n=32):
    nt = x.size(0)
    time = int(nt / n)
    x = x.view(time, n, *x.size()[1:])
    return x


def hard_sigmoid(x, alpha=0.0):
    return torch.clamp(x + 0.5, 0 - alpha, 1 + alpha)


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
        x = time_to_batch(x)
        x = self.module(x)
        x = batch_to_time(x, n)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class RNNCell(nn.Module):
    r"""
    defines minimum behavior of recurrent cell
    """
    def __init__(self, hard):
        super(RNNCell, self).__init__()
        self.sigmoid = hard_sigmoid if hard else torch.sigmoid
        self.tanh = F.hardtanh if hard else torch.tanh

    def detach_hidden(self):
        if isinstance(self.prev_hidden, list):
            for i in range(len(self.prev_hidden)):
                if self.prev_hidden[i] is not None:
                    self.prev_hidden[i] = self.prev_hidden[i].detach()
        elif self.prev_hidden is not None:
            self.prev_hidden = self.prev_hidden.detach()

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


class RNN(nn.Module):
    """
    base class doing the unrolling
    """

    def __init__(self, cell, hard=False):
        super(RNN, self).__init__()
        self.sigmoid = hard_sigmoid if hard else torch.sigmoid
        self.tanh = F.hardtanh if hard else torch.tanh
        self.cell = cell
        self.reset()

    def forward(self, x, alpha=1, future=0):
        self.detach_hidden()
        xseq = x.unbind(0)
        result = []
        ht = None

        #First treat sequence
        for t, xt in enumerate(xseq):
        	#this is a trick and should not be used in deterministic cases
            if ht is not None and torch.rand(1).item() > alpha and t > 1:
             	xt = ht.detach()
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
        for name, module in self.cell._modules.iteritems():
            if isinstance(module, RNNCell):
                module.detach_hidden()

    def reset(self, mask=None):
        for name, module in self.cell._modules.iteritems():
            if isinstance(module, RNNCell):
                module.reset(mask)


class LSTMCell(RNNCell):
    def __init__(self, in_channels, hidden_dim, x2h_func, h2h_func, hard=False, nonlinearity=torch.tanh):
        super(LSTMCell, self).__init__(hard)
        self.hidden_dim = hidden_dim
        self.x2h = x2h_func(in_channels, 4 * self.hidden_dim)
        self.h2h = h2h_func(self.hidden_dim, 4 * self.hidden_dim)
        self.act = nonlinearity
        self.prev_hidden = None
        self.reset()
        self.alpha = 1.0

    def forward(self, xt):
        if self.prev_hidden is None:
            prev_h, prev_c = None, None
        else:
            prev_h, prev_c = self.prev_hidden

        if self.prev_hidden is None:
            tmp = self.x2h(xt)
        else:
            tmp = self.x2h(xt) + self.h2h(prev_h)
            
        cc_i, cc_f, cc_o, cc_g = tmp.chunk(4, 1) #torch.split(tmp, self.hidden_dim, dim=1)
        f = torch.sigmoid(cc_f)
        i = torch.sigmoid(cc_i)
        o = torch.sigmoid(cc_o)
        g = self.act(cc_g)

        if self.prev_hidden:
            c = f * prev_c + i * g
        else:
            c = i * g

        h = o * self.act(c)
        self.prev_hidden = [h, c]
        return h



def lstm_fc(cin, cout):
    oph = partial(nn.Linear, bias=True)
    opx = partial(nn.Linear, bias=True)
    return LSTMCell(cin, cout, opx, oph, nonlinearity=torch.tanh)


def lstm_conv(cin, cout):
    oph = partial(nn.Conv2d, kernel_size=3, padding=1, stride=1, bias=True)
    opx = partial(nn.Conv2d, kernel_size=3, padding=1, stride=1, bias=True)
    return LSTMCell(cin, cout, opx, oph, nonlinearity=torch.tanh)



if __name__ == '__main__':
    import time

    t, n, c, h, w = 10, 4, 8, 64, 64
    cuda = True

    # Use-Case: Convolutional RNN
    x = torch.rand(t, n, c, h, w)
    conv_func = partial(Conv2d, kernel_size=5, padding=5//2, norm='weight')
    x2h_func = partial(conv_dw, kernel_size=3, norm='weight', stride=1)
    h2h_func = partial(nn.Conv2d, kernel_size=3, padding=1, stride=1)



    lstm = RNN(nn.Sequential(lstm_conv(c, 8),
                             lstm_conv(8, 8),
                             nn.Conv2d(8, c, kernel_size=3, padding=1, stride=1))
                )


    if cuda:
        x = x.cuda()
        lstm = lstm.cuda()

    start = time.time()
    for _ in range(1):
        lstm.reset()
        y = lstm(x)

    print(time.time()-start)