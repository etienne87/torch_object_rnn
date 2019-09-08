from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.ssd.box_coder import SSDBoxCoder
from core.ssd.loss import SSDLoss, FocalLoss
from core import modules as crnn
from core.modules import ConvBN, SequenceWise, time_to_batch, batch_to_time
import math


class FPN(nn.Module):
    def __init__(self, cin=1, cout=128, nmaps=3):
        super(FPN, self).__init__()
        self.cin = cin
        self.base = 8
        self.cout = cout
        self.nmaps = nmaps

        self.conv1 = SequenceWise(nn.Sequential(
            ConvBN(cin, self.base, kernel_size=7, stride=2, padding=3),
            ConvBN(self.base, self.base * 4, kernel_size=7, stride=2, padding=3)
        ))

        self.conv2 = crnn.UNet(self.base * 4,
                               self.cout * self.nmaps,
                               self.base * 4, 3,
                               up=crnn.UpCatRNN, down=crnn.down_ff)

        self.end_point_channels = [self.cout] * self.nmaps


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2 = time_to_batch(x2)[0]

        sources = torch.split(x2, self.cout, dim=1)
        return sources

    def reset(self):
        for name, module in self._modules.iteritems():
            if hasattr(module, "reset"):
                module.reset()


class Trident(nn.Module):
    def __init__(self, cin=1):
        super(Trident, self).__init__()
        self.cin = cin
        base = 8
        self.conv1 = ConvBN(cin, base, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBN(base, base * 4, kernel_size=7, stride=2, padding=3)

        self.conv3 = crnn.ConvRNN(base * 4, base * 8, kernel_size=7, stride=2, padding=3)
        self.conv4 = crnn.ConvRNN(base * 8, base * 8, kernel_size=7, stride=1, dilation=1, padding=3)
        self.conv5 = crnn.ConvRNN(base * 8, base * 8, kernel_size=7, stride=1, dilation=2, padding=3)
        self.conv6 = crnn.ConvRNN(base * 8, base * 8, kernel_size=7, stride=1, dilation=3, padding=3 * 2)

        self.end_point_channels = [self.conv3.cout,  # 8
                                   self.conv4.cout,  # 16
                                   self.conv5.cout,  # 32
                                   self.conv6.cout]  # 64

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

        x3, n = time_to_batch(x3)
        x4, n = time_to_batch(x4)
        x5, n = time_to_batch(x5)
        x6, n = time_to_batch(x6)

        sources += [x3, x4, x5, x6]
        return sources

    def reset(self):
        for name, module in self._modules.iteritems():
            if isinstance(module, crnn.ConvRNN):
                module.timepool.reset()


def get_box_params(sources, h, w):
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

        print("box size: ", box_sizes[-1])

    s_k = s_min + (s_max - s_min)
    box_sizes.append(s_k * image_size)

    return fm_sizes, steps, box_sizes


def init_prior_probability(shape, prior_probability=0.01):
    return torch.ones(shape) * -math.log((1 - prior_probability) / prior_probability)


def decode_boxes(box_map, num_classes, num_anchors):
    """
    box_map: N, C, H, W
    # N, C, H, W -> N, H, W, C -> N, HWNa, 4+Classes
    """
    fm_h, fm_w = box_map.shape[-2:]
    nboxes = fm_h * fm_w * num_anchors

    box_map = box_map.permute([0, 2, 3, 1]).contiguous().view(box_map.size(0), nboxes, 4 + num_classes)
    return box_map[..., :4], box_map[..., 4:]


class SSD(nn.Module):
    def __init__(self, feature_extractor=Trident,
                 num_classes=2, cin=2, height=300, width=300, act='softmax'):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.height, self.width = height, width
        self.cin = cin

        self.extractor = feature_extractor(cin)

        self.fm_sizes, self.steps, self.box_sizes = self.get_ssd_params()
        self.ary = float(width) / height

        self.aspect_ratios = []
        self.in_channels = self.extractor.end_point_channels
        self.num_anchors = [2 * len(self.aspect_ratios) + 2 for i in range(len(self.in_channels))]

        self.pred_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.pred_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i]*(4+self.num_classes),
                                           kernel_size=3, padding=1, stride=1)]

        self.act = act
        self.box_coder = SSDBoxCoder(self, 0.6, 0.4)
        self.criterion = SSDLoss(num_classes=num_classes)

    def get_ssd_params(self):
        x = Variable(torch.randn(1, 1, self.cin, self.height, self.width))
        sources = self.extractor(x)
        return get_box_params(sources, self.height, self.width)

    def reset(self):
        self.extractor.reset()

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        xs = self.extractor(x)
        for i, x in enumerate(xs):
            pred = self.pred_layers[i](x)
            loc_pred, cls_pred = decode_boxes(pred, self.num_classes, self.num_anchors[i])
            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))
            cls_preds.append(cls_pred.view(cls_pred.size(0), -1, self.num_classes))

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)

        if not self.training:
            if self.act == 'softmax':
                cls_preds = F.softmax(cls_preds, dim=2)
            else:
                cls_preds = torch.sigmoid(cls_preds)

        return loc_preds, cls_preds

    def compute_loss(self, x, targets):
        loc_preds, cls_preds = self(x)
        loc_targets, cls_targets = self.box_coder.encode_txn_boxes(targets)
        loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss = loc_loss + cls_loss
        return loss


if __name__ == '__main__':
    net = ConvRNNFeatureExtractor(1)

    t, n, c, h, w = 10, 1, 1, 128, 128
    x = torch.rand(t, n, c, h, w)

    y = net(x)