'''SSD model with custom feature extractor.'''
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from deform_conv import ConvOffset2D

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

    s_k = s_min + (s_max - s_min)
    box_sizes.append(s_k * image_size)

    return fm_sizes, steps, box_sizes


class SSD(nn.Module):
    def __init__(self, feature_extractor, num_classes=2, cin=2, height=300, width=300, act='softmax'):
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

        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()

        #shared heads
        #self.conv_offset = ConvOffset2D(self.in_channels[0])
        #self.loc_layer = nn.Conv2d(self.in_channels[0], self.num_anchors[0]*4, kernel_size=3, padding=1)
        #self.cls_layer = nn.Conv2d(self.in_channels[0], self.num_anchors[0]* self.num_classes, kernel_size=3, padding=1)

        for i in range(len(self.in_channels)):
            self.loc_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i]*4, kernel_size=3, padding=1)]
            self.cls_layers += [
                nn.Conv2d(self.in_channels[i], self.num_anchors[i] * self.num_classes, kernel_size=3, padding=1)]

        self.act = act

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
            loc_pred = self.loc_layers[i](x)
            #x = self.conv_offset(x)
            #loc_pred = self.loc_layer(x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))

            cls_pred = self.cls_layers[i](x)
            #cls_pred = self.cls_layer(x)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds.append(cls_pred.view(cls_pred.size(0), -1, self.num_classes))

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)

        if not self.training:
            if self.act == 'softmax':
                cls_preds = F.softmax(cls_preds, dim=2)
            else:
                cls_preds = torch.sigmoid(cls_preds)

        return loc_preds, cls_preds

