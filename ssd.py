'''SSD model with VGG16 as feature extractor.'''
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from coordconv import CoordConv

from models.torch_convlstm import ConvLSTM, batch_to_time, time_to_batch


class Conv2d(nn.Module):

    def __init__(self, cin, cout, kernel_size, stride, padding, dilation=1, addcoords=False):
        super(Conv2d, self).__init__()
        self.cin = cin
        self.cout = cout

        if addcoords:
            self.cin += 2
            self.conv1 = CoordConv(cin, cout, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                   padding=padding,
                                   bias=True)
        else:
            self.conv1 = nn.Conv2d(cin, cout, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                   padding=padding,
                                   bias=True)

        self.bn1 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        return x


def get_ssd_params(sources, h, w):
    image_size = float(min(h, w))
    steps = []
    box_sizes = []
    fm_sizes = []
    s_min, s_max = 0.1, 0.9
    m = float(len(sources))
    for k, src in enumerate(sources):
        # featuremap size
        fm_sizes.append((src.size(2), src.size(3)))

        # step is ratio image_size / featuremap_size
        step_y, step_x = math.floor(float(h) / src.size(2)), math.floor(float(w) / src.size(3))
        steps.append((step_y, step_x))

        # compute scale
        s_k = s_min + (s_max - s_min) * k / m

        # box_size is scale * image_size
        box_sizes.append(math.floor(s_k * image_size))

    s_k = s_min + (s_max - s_min)
    box_sizes.append(s_k * image_size)

    return fm_sizes, steps, box_sizes


class ConvRNNFeatureExtractor(nn.Module):
    def __init__(self, cin=1):
        super(ConvRNNFeatureExtractor, self).__init__()
        self.cin = cin
        base = 16
        _kernel_size = 7
        _stride = 2
        _padding = 3
        self.conv1 = Conv2d(cin, base, kernel_size=7, stride=2, padding=3, addcoords=False)
        self.conv2 = Conv2d(base, base * 2, kernel_size=7, stride=2, padding=3)
        self.conv3 = ConvLSTM(base * 2, base * 4, kernel_size=_kernel_size, stride=_stride, padding=_padding)
        self.conv4 = ConvLSTM(base * 4, base * 8, kernel_size=_kernel_size, stride=_stride, padding=_padding)
        self.conv5 = ConvLSTM(base * 8, base * 8, kernel_size=_kernel_size, stride=_stride, padding=_padding)
        self.conv6 = ConvLSTM(base * 8, base * 16, kernel_size=_kernel_size, stride=_stride, padding=_padding)

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
            # x3 = x3[:, :, -1]
            # x4 = x4[:, :, -1]
            # x5 = x5[:, :, -1]
            # x6 = x6[:, :, -1]
        sources += [x3, x4, x5, x6]
        return sources

    def get_ssd_params(self, h=300, w=300):
        x = Variable(torch.randn(1, self.cin, 1, h, w))
        sources = self(x)
        return get_ssd_params(sources, h, w)

    def reset(self):
        for name, module in self._modules.iteritems():
            if isinstance(module, ConvLSTM):
                module.timepool.reset()


class SSD(nn.Module):
    def __init__(self, num_classes, cin=2, height=300, width=300):
        super(SSD, self).__init__()
        self.num_classes = num_classes

        self.extractor = ConvRNNFeatureExtractor(cin)

        self.height, self.width = height, width
        self.fm_sizes, self.steps, self.box_sizes = self.extractor.get_ssd_params(height, width)
        self.ary = float(width) / height

        self.aspect_ratios = []
        self.in_channels = self.extractor.end_point_channels
        self.num_anchors = [2 * len(self.aspect_ratios) + 2 for i in range(len(self.in_channels))]

        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.loc_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i]*4, kernel_size=3, padding=1)]
            #self.loc_layers += [CoordConv(self.in_channels[i], self.num_anchors[i] * 4, kernel_size=3, padding=1)]
            self.cls_layers += [
                nn.Conv2d(self.in_channels[i], self.num_anchors[i] * self.num_classes, kernel_size=3, padding=1)]

    def reset(self):
        if isinstance(self.extractor, ConvRNNFeatureExtractor):
            self.extractor.reset()

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        xs = self.extractor(x)
        for i, x in enumerate(xs):
            loc_pred = self.loc_layers[i](x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))

            cls_pred = self.cls_layers[i](x)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds.append(cls_pred.view(cls_pred.size(0), -1, self.num_classes))

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        return loc_preds, cls_preds


if __name__ == '__main__':
    ssd = SSD(2, 240, 304)

