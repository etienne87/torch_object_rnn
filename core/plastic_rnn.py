from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.modules import RNNCell, SequenceWise, ConvLayer


def im2col(x, K, P):
    N, C, H, W = x.shape
    L = N * H * W
    CK2 = C * K ** 2
    xcol = F.unfold(x, kernel_size=K, padding=P)
    xcol = xcol.permute(0, 2, 1).reshape(L, CK2) 
    return xcol

def outer_conv(xcol, y):
    N, C, H, W = y.shape
    L, CK2 = xcol.shape
    y = hide_spatial(y) # [L, C]

    # [L, CK2, 1] x [L, 1, C] = [L, CK2, C]
    delta = torch.bmm(xcol.unsqueeze(2), y.unsqueeze(1)) 
    return delta

def local_convolution(self, xcol, w, out_shape):
    L, CK2 = xcol.shape
    N, C, H, W = out_shape
    y = torch.bmm(xcol.unsqueeze(1), w)
    return y

def show_spatial(x, N, H, W, C):
    return x.reshape(N, H, W, C).permute(0, 3, 1, 2).contiguous()

def hide_spatial(x):
    N, C, H, W = x.shape
    L = N*H*W
    return x.permute(0, 2, 3, 1).reshape(L, C)

def convolutions(x, weights, padding=1):
    return torch.cat([F.conv2d(x[i:i+1], weights[i], bias=None, padding=padding)  for i in range(len(x))])


class ConvPlastic(RNNCell):
    r"""ConvPlasticCell module, applies sequential part of ConvRNNPlastic.

    a convolutional derivation of https://arxiv.org/pdf/1804.02464.pdf

    the module learns weights by applying hebbian rule convolutionally.

    V1: the plastic weights are accumulated accross the receptive field (only global convolutions)
    """
    def __init__(self, in_channels, hidden_dim, kernel_size=3, hard=False, local=False):
        super(ConvPlastic, self).__init__(hard)
        self.hidden_dim = hidden_dim

        self.conv_x2h = SequenceWise(ConvLayer(in_channels, hidden_dim,
                                             kernel_size=5,
                                             stride=2,
                                             dilation=1,
                                             padding=2,
                                             activation='Identity'))

        self.K = kernel_size
        self.P = kernel_size//2
        self.C = hidden_dim
        self.CK2 = self.C * self.K**2

        # fixed part of weights 
        self.fixed_weights = nn.Parameter(.01 * torch.randn(self.C, self.C, self.K, self.K).float())

        # fixed modulation of plastic weights
        self.alpha = nn.Parameter(.01 * torch.randn(self.C, self.C, self.K, self.K).float())

        self.eta = nn.Parameter( .01 * torch.ones(1).float() )
        self.reset()

    def forward(self, xi):
        xi = self.conv_x2h(xi)
        xiseq = xi.split(1, 0)  # t,n,c,h,w

        T, N, C, H, W = xi.shape
        L = N*H*W

        if self.prev_h is not None:
            self.hebb = self.hebb.detach()
            self.prev_h = self.prev_h.detach()
        else:
            self.prev_h = torch.zeros(N, C, H, W).float().to(xi)
            self.hebb = torch.zeros(N, self.C, self.C, self.K, self.K).float().to(xi)

        result = []
        for t, xt in enumerate(xiseq):
            xt = xt.squeeze(0)
            self.prev_h, self.hebb = self.forward_t(xt, self.prev_h, self.hebb)
            result.append(self.prev_h.unsqueeze(0))
        
        res = torch.cat(result, dim=0)
        return res

    def forward_t(self, xt, hin, hebb):
        weights = hebb * self.alpha.unsqueeze(0) + self.fixed_weights
        hout = torch.tanh(convolutions(hin, weights) + xt)
        hebb = self.update_hebbian(hin, hout, hebb)
        return hout, hebb

    def update_hebbian(self, hin, hout, hebb):
        N, C, H, W = hout.shape
        hin_col = im2col(hin, self.K, self.P)
        delta = outer_conv(hin_col, hout)
        delta = delta.reshape(N, H*W, self.CK2, C).mean(dim=1) # [N, CK2, C]
        delta = delta.reshape(N, C,  self.K, self.K, self.C).permute(0, 1, 4, 2, 3) # [N, C, C, K, K]
        hebb = (hebb + self.eta * delta).clamp_(-1, 1)
        return hebb 
    
    def reset(self):
        self.prev_h = None
        self.hebb = None


if __name__ == '__main__':
    x = torch.rand(10, 5, 7, 16, 16)

   
    net = ConvPlasticCell(7)
    y = net(x)
    print(y.shape)