from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


from core.ssd.loss import SSDLoss
from core.backbones import FPN
from core.anchors import Anchors
import math


#TODO: add a "RPN" module with class BoxHead(num_anchors, num_classes, act)

class SingleStageDetector(nn.Module):
    def __init__(self, feature_extractor=FPN,
                 num_classes=2, cin=2, act='sigmoid'):
        super(SingleStageDetector, self).__init__()
        self.num_classes = num_classes
        self.cin = cin

        self.feature_extractor = feature_extractor(cin)

        self.box_coder = Anchors(pyramid_levels=[i for i in range(3,3+self.feature_extractor.levels)],
                                 scales=[1.0, 1.5],
                                 ratios=[1],
                                 label_offset=1,
                                 fg_iou_threshold=0.5, bg_iou_threshold=0.4)

        self.num_anchors = self.box_coder.num_anchors

        self.aspect_ratios = []
        self.act = act

        self.loc_head = self._make_head(self.feature_extractor.cout, self.num_anchors * 4)
        self.cls_head = self._make_head(self.feature_extractor.cout, self.num_anchors * self.num_classes)

        torch.nn.init.normal_(self.loc_head[-1].weight, std=0.01)
        torch.nn.init.constant_(self.loc_head[-1].bias, 0)

        if self.act == 'softmax':
            self.softmax_init(self.cls_head[-1])
        else:
            self.sigmoid_init(self.cls_head[-1])

        self.criterion = SSDLoss(num_classes=num_classes,
                                 mode='focal',
                                 use_sigmoid=self.act=='sigmoid',
                                 use_iou=False)

    def sigmoid_init(self, l):
        px = 0.99
        bias_bg = math.log(px / (1 - px))
        torch.nn.init.normal_(l.weight, std=0.01)
        torch.nn.init.constant_(l.bias, 0)
        l.bias.data = l.bias.data.reshape(self.num_anchors, self.num_classes)
        l.bias.data[:, 0:] -= bias_bg
        l.bias.data = l.bias.data.reshape(-1)

    def softmax_init(self, l):
        px = 0.99
        K = self.num_classes - 1
        bias_bg = math.log(K * px / (1 - px))
        torch.nn.init.normal_(l.weight, std=0.01)
        torch.nn.init.constant_(l.bias, 0)
        l.bias.data = l.bias.data.reshape(self.num_anchors, self.num_classes)
        l.bias.data[:, 0] += bias_bg
        l.bias.data = l.bias.data.reshape(-1)

    def _make_head(self, in_planes, out_planes):
        layers = []
        layers.append(nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(True))

        for _ in range(0):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))

        return nn.Sequential(*layers)

    def reset(self):
        self.feature_extractor.reset()

    def _apply_head(self, layer, xs, ndims):
        out = []
        for x in xs:
            y = layer(x).permute(0, 2, 3, 1).contiguous()
            y = y.view(y.size(0), -1, ndims)
            out.append(y)
        out = torch.cat(out, 1)
        return out

    def _forward_shared(self, xs):
        loc_preds = self._apply_head(self.loc_head, xs, 4)
        cls_preds = self._apply_head(self.cls_head, xs, self.num_classes)

        if not self.training:
            if self.act == 'softmax':
                cls_preds = F.softmax(cls_preds, dim=2)
            else:
                cls_preds = torch.sigmoid(cls_preds)

        return loc_preds, cls_preds


    def forward(self, x):
        xs = self.feature_extractor(x)
        return self._forward_shared(xs)

    def compute_loss(self, x, targets):
        xs = self.feature_extractor(x)
        loc_preds, cls_preds = self._forward_shared(xs)

        with torch.no_grad():
            loc_targets, cls_targets = self.box_coder.encode(xs, targets)

        loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss_dict = {'loc': loc_loss, 'cls_loss': cls_loss}

        return loss_dict

    def get_boxes(self, x, score_thresh=0.4):
        xs = self.feature_extractor(x)
        loc_preds, cls_preds = self._forward_shared(xs)
        targets = self.box_coder.decode(xs, loc_preds, cls_preds, x.size(1), score_thresh=score_thresh)
        return targets
