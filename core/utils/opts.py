from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
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
