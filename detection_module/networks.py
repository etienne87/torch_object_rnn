import torch.nn as nn
from modules import Conv2d, ConvLSTM, ConvGRU, batch_to_time, time_to_batch
from torch.nn import functional as F

class ConvRNNFeatureExtractor(nn.Module):
    def __init__(self, cin=1):
        super(ConvRNNFeatureExtractor, self).__init__()
        self.cin = cin
        base = 8
        _kernel_size = 7
        _stride = 2
        _padding = 3
        self.conv1 = Conv2d(cin, base, kernel_size=7, stride=2, padding=3, addcoords=False)
        self.conv2 = Conv2d(base, base * 2, kernel_size=7, stride=2, padding=3)
        self.conv3 = ConvGRU(base * 2, base * 4, kernel_size=_kernel_size, stride=_stride, padding=_padding)
        self.conv4 = ConvGRU(base * 4, base * 8, kernel_size=_kernel_size, stride=_stride, padding=_padding)
        self.conv5 = ConvGRU(base * 8, base * 8, kernel_size=_kernel_size, stride=_stride, padding=_padding)
        self.conv6 = ConvGRU(base * 8, base * 16, kernel_size=_kernel_size, stride=_stride, padding=_padding)

        self.end_point_channels = [self.conv3.cout,  # 8
                                   self.conv4.cout,  # 16
                                   self.conv5.cout,  # 32
                                   self.conv6.cout]  # 64

        self.return_all = True #if set returns rank-5, else returns rank-4 last item

    def forward(self, x):
        sources = list()

        x0, n = time_to_batch(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x2 = batch_to_time(x2, n)

        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        if self.return_all:
            x3, n = time_to_batch(x3)
            x4, n = time_to_batch(x4)
            x5, n = time_to_batch(x5)
            x6, n = time_to_batch(x6)
        else:
            x3 = x3[-1]
            x4 = x4[-1]
            x5 = x5[-1]
            x6 = x6[-1]
        sources += [x3, x4, x5, x6]
        return sources

    def reset(self):
        for name, module in self._modules.iteritems():
            if isinstance(module, ConvLSTM) or \
               isinstance(module, ConvGRU) or \
               isinstance(module, ConvQRNN):
                module.timepool.reset()


class FPNRNNFeatureExtractor(nn.Module):
    def __init__(self, cin=1):
        super(FPNRNNFeatureExtractor, self).__init__()
        self.cin = cin
        base = 4
        _kernel_size = 7
        _stride = 2
        _padding = 3
        self.conv1 = Conv2d(cin, base, kernel_size=7, stride=2, padding=3, addcoords=False)
        self.conv2 = Conv2d(base, base * 2, kernel_size=7, stride=2, padding=3)
        self.conv3 = Conv2d(base * 2, base * 4, kernel_size=7, stride=2, padding=3)
        self.conv4 = ConvLSTM(base * 4, base * 8, kernel_size=_kernel_size, stride=_stride, padding=_padding)
        self.conv5 = Conv2d(base * 8, base * 8, kernel_size=3, stride=1, padding=1)
        self.conv6 = Conv2d(base * 8, base * 8, kernel_size=3, stride=1, padding=1)
        self.conv35 = Conv2d(base * 4, base * 8, kernel_size=1, stride=1, padding=0)
        self.conv26 = Conv2d(base * 2, base * 8, kernel_size=1, stride=1, padding=0)
        self.conv7 = Conv2d(base * 8, base * 8, kernel_size=1, stride=1, padding=0)
        self.conv8 = Conv2d(base * 8, base * 8, kernel_size=1, stride=1, padding=0)

        self.end_point_channels = [self.conv5.cout,  # 32
                                   self.conv6.cout,
                                   self.conv7.cout,
                                   self.conv8.cout]# 64

        self.return_all = True #if set returns rank-5, else returns rank-4 last item

    def forward(self, x):
        sources = list()

        x0, n = time_to_batch(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x3b = batch_to_time(x3, n)
        x4, n = time_to_batch(self.conv4(x3b))
        x35 = self.conv35(x3)
        x26 = self.conv26(x2)
        x5 = self.conv5(F.interpolate(x4, scale_factor=2) + x35)
        x6 = self.conv6(F.interpolate(x5, scale_factor=2) + x26)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)

        if not self.return_all:
            x5 = batch_to_time(x5, n)[-1]
            x6 = batch_to_time(x6, n)[-1]
            x7 = batch_to_time(x7, n)[-1]
            x8 = batch_to_time(x8, n)[-1]
        sources += [x5, x6, x7, x8]
        return sources

    def reset(self):
        for name, module in self._modules.iteritems():
            if isinstance(module, ConvLSTM) or isinstance(module, ConvGRU):
                module.timepool.reset()

