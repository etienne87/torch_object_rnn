from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


from core.losses import DetectionLoss
from core.backbones import FPN, MobileNetFPN
from core.anchors import Anchors
from core.utils import box
from core.rpn import BoxHead
from core.fpn import FeaturePyramidNetwork



class RefinedDetector(nn.Module):
    def __init__(self, feature_extractor=MobileNetFPN, rpn=BoxHead,
                 num_classes=2, cin=2, act='sigmoid'):
        super(RefinedDetector, self).__init__()
        self.label_offset = 1 * (act=='softmax')
        self.num_classes = num_classes
        self.cin = cin

        self.feature_extractor = feature_extractor(cin)

        self.box_coder = Anchors(pyramid_levels=[i for i in range(3,3+self.feature_extractor.levels)],
                                 scales=[1.0, 2**1./3, 2**2./3],
                                 ratios=[0.5, 1.0, 2.0],
                                 fg_iou_threshold=0.5, bg_iou_threshold=0.4)

        self.num_anchors = self.box_coder.num_anchors
        self.act = act

        self.rpn = rpn(self.feature_extractor.cout, self.box_coder.num_anchors, self.num_classes + self.label_offset, act)

        # refinement
        self.feature_extractor2 = FeaturePyramidNetwork([self.feature_extractor.cout]*self.feature_extractor.levels,
                                                      self.feature_extractor.cout)
        self.rpn2 = rpn(self.feature_extractor.cout, self.box_coder.num_anchors, self.num_classes + self.label_offset,
                       act)

        self.criterion = DetectionLoss('sigmoid_focal_loss')

    def reset(self):
        self.feature_extractor.reset()

    def forward(self, x):
        xs = self.feature_extractor(x)
        ys = self.feature_extractor2(xs)
        return self.rpn(xs), self.rpn(ys)

    def compute_loss(self, x, targets):
        xs = self.feature_extractor(x)
        ys = self.feature_extractor2(xs)
        loc_preds, cls_preds = self.rpn(xs)
        loc_preds2, cls_preds2 = self.rpn(xs)

        with torch.no_grad():
            loc_targets, cls_targets = self.box_coder.encode(xs, targets)
        with torch.no_grad():
            anchors, anchors_xyxy = self.box_coder(xs)
            anchors2 = box.deltas_to_bbox(loc_preds, anchors)
            anchors2xyxy = box.change_box_order(anchors2, 'xywh2xyxy')
            loc_targets2, cls_targets2 = self.box_coder.encode_with_anchors(anchors2, anchors2xyxy, targets)
            #cls_targets *= cls_preds?


        assert cls_targets.shape[1] == cls_preds.shape[1]
        loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loc_loss2, cls_loss2 = self.criterion(loc_preds2, loc_targets2, cls_preds2, cls_targets2)
        loss_dict = {'loc': loc_loss, 'cls_loss': cls_loss,
                     'loc2': loc_loss2, 'cls_loss2': cls_loss2}

        return loss_dict

    def get_refined_anchors(self, xs, loc_preds):
        anchors, _ = self.box_coder(xs)
        anchors2 = box.deltas_to_bbox(loc_preds, anchors)
        anchors2xyxy = box.change_box_order(anchors2, 'xywh2xyxy')
        return anchors2, anchors2xyxy

    def get_boxes(self, x, score_thresh=0.4):
        xs = self.feature_extractor(x)
        ys = self.feature_extractor2(xs)
        loc_preds, cls_preds = self.rpn2(ys)
        loc_preds2, cls_preds2 = self.rpn2(ys)

        anchors, anchorsxyxy = self.get_refined_anchors(xs, loc_preds)
        scores = cls_preds2[..., self.label_offset:].contiguous()
        targets = self.box_coder.decode_with_anchors(anchors, anchorsxyxy, loc_preds, scores, x.size(1), score_thresh=score_thresh)
        return targets
