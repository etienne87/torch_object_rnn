from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.ops.poolers as pool
from core.losses import DetectionLoss
from core.backbones import FPN
from core.anchors import Anchors
from core.rpn import BoxHead


class TwoStageDetector(nn.Module):
    def __init__(self, feature_extractor=FPN, rpn=BoxHead,
                 num_classes=2, cin=2, act='sigmoid'):
        super(TwoStageDetector, self).__init__()
        self.label_offset = 1 * (act=='softmax')
        self.num_classes = num_classes
        self.cin = cin

        self.feature_extractor = feature_extractor(cin)

        self.box_coder = Anchors(pyramid_levels=[i for i in range(3,3+self.feature_extractor.levels)],
                                 scales=[1.0, 1.5],
                                 ratios=[1],
                                 fg_iou_threshold=0.5, bg_iou_threshold=0.4)

        self.num_anchors = self.box_coder.num_anchors
        self.act = act

        self.rpn = rpn(self.feature_extractor.cout, self.box_coder.num_anchors, 1 + self.label_offset, act)

        self.roi_pool = pool.MultiScaleRoIAlign(feat_names, output_size)
        self.second_stage = rpn(self.feature_extractor.cout,
                                         self.box_coder.num_anchors,
                                         self.num_classes + self.label_offset, act)


        self.criterion = DetectionLoss('sigmoid_focal_loss')



    def reset(self):
        self.feature_extractor.reset()

    def forward(self, x):
        xs = self.feature_extractor(x)
        loc_preds, cls_preds = self.rpn(xs)

        scores = cls_preds[..., self.label_offset:].contiguous()
        rois = self.box_coder.decode(xs, loc_preds, scores, x.size(1), score_thresh=score_thresh)

        out = self.roi_pool(xs, rois)

        loc_rois, cls_rois = self.second_stage(out)

        #decode loc_rois knowing rois using bbox_to_deltas


        return boxes

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
        scores = cls_preds[..., self.label_offset:].contiguous()
        targets = self.box_coder.decode(xs, loc_preds, scores, x.size(1), score_thresh=score_thresh)
        return targets