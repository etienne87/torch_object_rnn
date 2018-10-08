'''SSD model with VGG16 as feature extractor.'''
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from coordconv import CoordConv
import sys

sys.path.insert(0, '../')
from models.torch_convlstm import ConvLSTM, ConvQRNN


def _time_to_batch(x):
    n, c, t, h, w = x.size()
    x = x.permute([0, 2, 1, 3, 4]).contiguous().view(n * t, c, h, w)
    return x, n


def _batch_to_time(x, n=32):
    nt, c, h, w = x.size()
    t = int(nt / n)
    x = x.view(n, t, c, h, w).permute([0, 2, 1, 3, 4]).contiguous()
    return x


class Conv2d(nn.Module):

    def __init__(self, nInputPlane, nOutputPlane, kernel_size, stride, padding, dilation=1, addcoords=False):
        super(Conv2d, self).__init__()
        self.Cin = nInputPlane
        self.Cout = nOutputPlane

        if addcoords:
            self.Cin += 2
            self.conv1 = CoordConv(nInputPlane, nOutputPlane, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                   padding=padding,
                                   bias=True)
        else:
            self.conv1 = nn.Conv2d(nInputPlane, nOutputPlane, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                   padding=padding,
                                   bias=True)

        self.bn1 = nn.BatchNorm2d(nOutputPlane)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        return x


class L2Norm(nn.Module):
    '''L2Norm layer across all channels.'''

    def __init__(self, in_features, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters(scale)

    def reset_parameters(self, scale):
        nn.init.constant(self.weight, scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        scale = self.weight[None, :, None, None]
        return scale * x


class FeatureExtractor(nn.Module):
    def __init__(self, nInputPlane=1):
        super(FeatureExtractor, self).__init__()
        self.nInputPlane = nInputPlane
        base = 2
        self.conv1 = Conv2d(nInputPlane, base, kernel_size=7, stride=2, padding=3, addcoords=True)
        self.conv2 = Conv2d(base, base * 2, kernel_size=7, stride=2, padding=3)
        self.conv3 = Conv2d(base * 2, base * 4, kernel_size=7, stride=2, padding=3)
        self.conv4 = Conv2d(base * 4, base * 8, kernel_size=7, stride=2, padding=3)
        self.conv5 = Conv2d(base * 8, base * 8, kernel_size=7, stride=2, padding=3)
        self.conv6 = Conv2d(base * 8, base * 16, kernel_size=7, stride=2, padding=3)

        self.end_point_channels = [self.conv3.Cout,  # 8
                                   self.conv4.Cout,  # 16
                                   self.conv5.Cout,  # 32
                                   self.conv6.Cout]  # 64

    def forward(self, x):
        sources = list()
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        sources += [x3, x4, x5, x6]
        return sources

    def get_ssd_params(self, w=300, h=300):
        x = Variable(torch.randn(1, self.nInputPlane, h, w))
        sources = self(x)
        return self._get_ssd_params(sources, w, h)

    @staticmethod
    def _get_ssd_params(sources, h, w):
        image_size = float(min(h, w))
        steps = []
        box_sizes = []
        fm_sizes = []
        s_min, s_max = 0.1, 0.9
        m = float(len(sources))
        for k, src in enumerate(sources):
            # featuremap size
            # f = min(src.size(2),src.size(3))
            # fm_sizes.append(f)
            fm_sizes.append((src.size(2), src.size(3)))

            # step is ratio image_size / featuremap_size
            # step = image_size / f
            # steps.append(math.floor(step))
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
    def __init__(self, nInputPlane=1):
        super(ConvRNNFeatureExtractor, self).__init__()
        self.nInputPlane = nInputPlane
        base = 16
        _kernel_size = (1, 7, 7)
        _stride = (1, 2, 2)
        _padding = (0, 3, 3)
        self.conv1 = Conv2d(nInputPlane, base, kernel_size=7, stride=2, padding=3, addcoords=True)
        self.conv2 = Conv2d(base, base * 2, kernel_size=7, stride=2, padding=3)
        self.conv3 = ConvLSTM(base * 2, base * 4, kernel_size=_kernel_size, stride=_stride, padding=_padding)
        self.conv4 = ConvLSTM(base * 4, base * 8, kernel_size=_kernel_size, stride=_stride, padding=_padding)
        self.conv5 = ConvLSTM(base * 8, base * 8, kernel_size=_kernel_size, stride=_stride, padding=_padding)
        self.conv6 = ConvLSTM(base * 8, base * 16, kernel_size=_kernel_size, stride=_stride, padding=_padding)

        self.end_point_channels = [self.conv3.Cout,  # 8
                                   self.conv4.Cout,  # 16
                                   self.conv5.Cout,  # 32
                                   self.conv6.Cout]  # 64

        self.return_all = False

    def forward(self, x):
        sources = list()

        x0, n = _time_to_batch(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        #x2 = F.avg_pool2d(x2, 2, 2)
        x2 = _batch_to_time(x2, n)

        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        if self.return_all:
            x3, n = _time_to_batch(x3)
            x4, n = _time_to_batch(x4)
            x5, n = _time_to_batch(x5)
            x6, n = _time_to_batch(x6)
        else:
            x3 = x3[:, :, -1]
            x4 = x4[:, :, -1]
            x5 = x5[:, :, -1]
            x6 = x6[:, :, -1]
        sources += [x3, x4, x5, x6]
        return sources

    def get_ssd_params(self, h=300, w=300):
        x = Variable(torch.randn(1, self.nInputPlane, 1, h, w))
        sources = self(x)
        return FeatureExtractor._get_ssd_params(sources, h, w)

    def reset(self):
        for name, module in self._modules.iteritems():
            if isinstance(module, ConvQRNN) or isinstance(module, ConvLSTM):
                module.timepool.reset()


class SSD300(nn.Module):
    def __init__(self, num_classes, in_channels=2, height=300, width=300):
        super(SSD300, self).__init__()
        self.num_classes = num_classes

        self.extractor = ConvRNNFeatureExtractor(in_channels)
        # self.extractor = FeatureExtractor(in_channels)

        self.height, self.width = height, width
        self.fm_sizes, self.steps, self.box_sizes = self.extractor.get_ssd_params(height, width)

        self.ary = float(width) / height
        # debug
        # print('fmsizes:',self.fm_sizes)
        # print('steps:',self.steps)
        # print('box_sizes:',self.box_sizes)

        self.aspect_ratios = []
        self.in_channels = self.extractor.end_point_channels
        self.num_anchors = [2 * len(self.aspect_ratios) + 2 for i in range(len(self.in_channels))]

        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            # self.loc_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i]*4, kernel_size=3, padding=1)]
            self.loc_layers += [CoordConv(self.in_channels[i], self.num_anchors[i] * 4, kernel_size=3, padding=1)]
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
    ssd = SSD300(2, 240, 304)

