from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from core.ssd.loss import SSDLoss
from core.losses import DetectionLoss

from core.backbones import FPN
from core.anchors import Anchors
from core.rpn import BoxHead



class SingleStageDetector(nn.Module):
    def __init__(self, feature_extractor=FPN, rpn=BoxHead,
                 num_classes=2, cin=2, act='sigmoid'):
        super(SingleStageDetector, self).__init__()
        self.label_offset = 1 * (act=='softmax')
        self.num_classes = num_classes + self.label_offset
        self.cin = cin

        self.feature_extractor = feature_extractor(cin)

        self.box_coder = Anchors(pyramid_levels=[i for i in range(3,3+self.feature_extractor.levels)],
                                 scales=[1.0, 1.5],
                                 ratios=[1],
                                 label_offset=self.label_offset,
                                 fg_iou_threshold=0.5, bg_iou_threshold=0.4)

        self.num_anchors = self.box_coder.num_anchors
        self.act = act

        self.rpn = rpn(self.feature_extractor.cout, self.box_coder.num_anchors, self.num_classes, act)

        # self.criterion = SSDLoss(num_classes=self.num_classes,
        #                          mode='focal',
        #                          use_sigmoid=self.act=='sigmoid',
        #                          use_iou=False)

        self.criterion = DetectionLoss()

    def reset(self):
        self.feature_extractor.reset()

    def forward(self, x):
        xs = self.feature_extractor(x)
        return self.rpn(xs)

    def compute_loss(self, x, targets):
        xs = self.feature_extractor(x)
        loc_preds, cls_preds = self.rpn(xs)

        with torch.no_grad():
            loc_targets, cls_targets = self.box_coder.encode(xs, targets)

        loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss_dict = {'loc': loc_loss, 'cls_loss': cls_loss}

        return loss_dict

    def get_boxes(self, x, score_thresh=0.4):
        xs = self.feature_extractor(x)
        loc_preds, cls_preds = self.rpn(xs)
        targets = self.box_coder.decode(xs, loc_preds, cls_preds, x.size(1), score_thresh=score_thresh)
        return targets
