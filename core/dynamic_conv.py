from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


def conv_dynamic(x, weights, biases, s, p, d):
    """
    conv with image-specific weights

    param x: N,Ci,H,W
    param weights: N,Co,Ci,K,K 
    param biases: N,Co 
    """
    n,ci,h,w = x.shape
    n,co,ci,k,k = weights.shape
    xgrouped = x.reshape(1, n * ci, h, w)
    wgrouped = weights.reshape(co * n, ci, k, k)
    y = F.conv2d(xgrouped, wgrouped, None, s, p, d, groups=n)
    y = y.reshape(n, co, h, w)
    return y + biases


def conv_dynamic_debug(x, weights, biases, s, p, d):
    y = torch.cat([F.conv2d(x[i:i+1], weights[i], None, s, p, d, 1) for i in range(x.shape[0])])
    return y + biases



class DynamicConv(nn.Module):
    """Attention-over-weights
    
    Dynamic Convolution: Attention over Convolution Kernel 
    https://arxiv.org/pdf/1912.03458.pdf       

    NOT SUPPORTING GROUPS YET 
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, 
                    num_weights=4, temperature=30):
        super(DynamicConv, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weights = nn.Parameter(torch.randn(out_channels, num_weights, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.biases = nn.Parameter(torch.zeros(out_channels, num_weights, 1, 1), requires_grad=True)

        self.att_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // num_weights, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // num_weights, num_weights, kernel_size=1)
        )
        self.tau = temperature

        for i in range(num_weights):
            init.kaiming_normal_(self.weights[i:i+1], mode='fan_out', nonlinearity='sigmoid')
       

    def forward(self, x):
        pi = F.avg_pool2d(self.att_fc(x), x.size(2)) #n,m,1,1
        pi = F.softmax(pi/self.tau, dim=1)[:,None,:,:,:] #n,1,m,1,1,1

        weights = torch.sum(self.weights[None] * pi[...,None], dim=2) #1,co,m,ci,k,k * n,1,m,1,1,1
        biases = torch.sum(self.biases[None] * pi, dim=2) #1,co,m,1,1 * n,1,m,1,1

        y = conv_dynamic(x, weights, biases, self.stride, self.padding, self.dilation)
        return y


def check_theoric():
    w = torch.rand(3,32,16,3,3)
    b = torch.rand(3,32,1,1)
    y = conv_dynamic(x, w, b, 1, 1, 1)
    y2 = conv_dynamic_debug(x, w, b, 1, 1, 1)
    diff = (y-y2).abs().max()
    print('diff: ', diff)


if __name__ == '__main__':
    x = torch.randn(3,16,32,32)


    net = DynamicConv(16, 32, 3, 1, 1)
    y = net(x)

    #print(y)