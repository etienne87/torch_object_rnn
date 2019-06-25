import torch.nn as nn
from modules import Conv2d, ConvLSTM, ConvGRU, batch_to_time, time_to_batch


class ConvRNNFeatureExtractor(nn.Module):
    def __init__(self, cin=1):
        super(ConvRNNFeatureExtractor, self).__init__()
        self.cin = cin
        base = 8
        self.conv1 = Conv2d(cin, base, kernel_size=7, stride=2, padding=3, addcoords=False)
        self.conv2 = Conv2d(base, base * 4, kernel_size=7, stride=2, padding=3)
        
        self.conv3 = ConvLSTM(base * 4, base * 8, kernel_size=7, stride=2, padding=3)
        self.conv4 = ConvLSTM(base * 8, base * 16, kernel_size=7, stride=1, dilation=1, padding=3)
        self.conv5 = ConvLSTM(base * 16, base * 16, kernel_size=7, stride=1, dilation=2, padding=3)
        self.conv6 = ConvLSTM(base * 16, base * 16, kernel_size=7, stride=1, dilation=3, padding=3*2)

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
               isinstance(module, ConvGRU):
                module.timepool.reset()

