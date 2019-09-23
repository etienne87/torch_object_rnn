from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import glob


def time_to_batch(x):
    t, n = x.size()[:2]
    x = x.view(n * t, *x.size()[2:])
    return x, n

def batch_to_time(x, n=32):
    nt = x.size(0)
    time = nt // n
    x = x.view(time, n, *x.size()[1:])
    return x

def load_last_checkpoint(net, logdir):
    checkpoints = glob.glob(logdir + '/checkpoints/' + '*.pth')
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('checkpoint#')[1].split('.pth')[0]))
    last_checkpoint = checkpoints[-1]
    checkpoint = torch.load(last_checkpoint)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    return start_epoch


class BatchRenorm(nn.Module):
    r"""
    BatchRenorm
    https://papers.nips.cc/paper/6790-batch-renormalization-towards-reducing-minibatch-dependence-in-batch-normalized-models.pdf
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(BatchRenorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.FloatTensor(1,num_features,1,1))
            self.bias = nn.Parameter(torch.FloatTensor(1,num_features,1,1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(1,num_features,1,1))
        self.register_buffer('running_var', torch.ones(1,num_features,1,1))
        self.reset_parameters()
        self.rmax = 1
        self.dmax = 0

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()
        self.rmax = 1
        self.dmax = 0

    def forward(self, input_):
        x = input_.permute(0, 2, 3, 1).contiguous().view(-1, self.num_features)
        mean = x.mean(dim=0)[None,:,None,None]
        var = x.var(dim=0)[None,:,None,None] + self.eps
        std = torch.sqrt(var)
        y = (input_ - mean)/std

        r = (mean / self.running_mean).data.clamp_(1./self.rmax, self.rmax)
        d = torch.sqrt(var / self.running_var).data.clamp_(-self.dmax, self.dmax)
        out = y * r * self.weight + d + self.bias

        self.running_mean += self.momentum * (self.running_mean - mean)
        self.running_var += self.momentum * (self.running_var - var)

        self.rmax += 1e-3
        self.dmax += 1e-3
        return out

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' max_length={max_length}, affine={affine})'
            .format(name=self.__class__.__name__, **self.__dict__))

