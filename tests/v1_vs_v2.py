"""
test v1 vs v2 code
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division 

import torch
import torch.nn as nn
from core.utils.opts import time_to_batch, batch_to_time
from core.modules import ConvLayer, SequenceWise, ConvRNN, Bottleneck, BottleneckLSTM
from core.unet import UNet

from core.modules_v2 import RNNWise, ConvLSTM
from core.unet_v2 import UNet as UNetV2
from functools import partial 



class TestBatchVsSequential(object):    
    def pytestcase_conv_seq_vs_parallel(self):
        t, n, c, h, w = 6, 3, 32, 32, 32
        x = torch.rand(t, n, c, h, w)
        m = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.InstanceNorm2d(c))
        m1 = SequenceWise(m, parallel=True)
        m2 = SequenceWise(m, parallel=False)
        y1 = m1(x)
        y2 = m2(x)
        assert (y1-y2).abs().max().item() == 0
    
    def pytestcase_bottleneck_seq_vs_parallel(self):
        t, n, c, h, w = 6, 3, 32, 32, 32
        x = torch.rand(t, n, c, h, w)
        m = nn.Sequential(Bottleneck(c, 2*c))
        m1 = SequenceWise(m, parallel=True)
        m2 = SequenceWise(m, parallel=False)
        y1 = m1(x)
        y2 = m2(x)
        assert (y1-y2).abs().max().item() == 0
    
    def pytestcase_net_seq_vs_parallel(self):
        t, n, c, h, w = 6, 3, 32, 32, 32
        x = torch.rand(t, n, c, h, w)
        base = 8
        m = nn.Sequential(Bottleneck(c, base * 2, 2),
        Bottleneck(base * 2, base * 4, 2),
        Bottleneck(base * 4, base * 8, 2))
        m1 = SequenceWise(m, parallel=True)
        m2 = SequenceWise(m, parallel=False)
        y1 = m1(x)
        y2 = m2(x)
        assert (y1-y2).abs().max().item() == 0

    def pytestcase_convrnn_vs_convlstm(self):
        t, n, c, h, w = 6, 3, 32, 32, 32
        x = torch.rand(t, n, c, h, w)
        m1 = ConvRNN(c, 2*c)
        m2 = ConvLSTM(c, 2*c)
        m2.conv_x2h[0].weight.data[...] = m1.conv_x2h.module[0].weight.data
        m2.conv_x2h[0].bias.data[...] = m1.conv_x2h.module[0].bias.data  
        m2.conv_h2h.weight.data[...] = m1.timepool.conv_h2h.weight.data
        m2.conv_h2h.bias.data[...] = m1.timepool.conv_h2h.bias.data
        m2 = RNNWise(m2)
        y1 = m1(x)
        y2 = m2(x)
        assert (y1-y2).abs().max().item() == 0


    """ def pytestcase_unet_v1_vs_unet_v2(self):
        t, n, c, h, w = 6, 3, 3, 32, 32
        x = torch.rand(t, n, c, h, w)
        m1 = V1(3)
        m2 = V2(3)
        y1 = m1(x)
        y2 = m2(x)
        for a, b in zip(y1, y2):
            print(a.shape, b.shape)
            import pdb;pdb.set_trace()
            diff = (a-b).abs().max().item()
            print(diff) """
            #assert (a-b).abs().max().item() == 0 
    
    
