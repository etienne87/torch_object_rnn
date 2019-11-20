from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


from core.losses import DetectionLoss
from core.backbones import FPN, MobileNetFPN, ResNet50FPN
from core.anchors import Anchors
from core.rpn import BoxHead, SSDBoxHead



class SingleStageDetector(nn.Module):
    def __init__(self, feature_extractor=FPN, rpn=BoxHead,
                 in_channels=3, 
                 num_classes=2,  
                 act='sigmoid', 
                 ratios=[0.5,1.0,2.0], 
                 scales=[1.0,2**1./3,2**2./3], 
                 nlayers=3,
                 loss='_focal_loss'):
        super(SingleStageDetector, self).__init__()
        self.label_offset = 1 * (act=='softmax')
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.feature_extractor = feature_extractor(in_channels)

        self.box_coder = Anchors(pyramid_levels=[i for i in range(3,3+self.feature_extractor.levels)],
                                 scales=scales,
                                 ratios=ratios,
                                 fg_iou_threshold=0.5, bg_iou_threshold=0.4)

        self.num_anchors = self.box_coder.num_anchors
        self.act = act

        self.rpn = rpn(self.feature_extractor.cout, self.box_coder.num_anchors, self.num_classes + self.label_offset, act, nlayers)

        self.criterion = DetectionLoss(act + loss) 

    def reset(self):
        self.feature_extractor.reset()

    def forward(self, x):
        xs = self.feature_extractor(x)
        return self.rpn(xs)

    def compute_loss(self, x, targets):
        xs = self.feature_extractor(x)
        loc_preds, cls_preds = self.rpn(xs)

        with torch.no_grad():
            anchors, anchors_xyxy = self.box_coder(xs)
            loc_targets, cls_targets = self.box_coder.encode(anchors, anchors_xyxy, targets)

        assert cls_targets.shape[1] == cls_preds.shape[1]
        loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss_dict = {'loc': loc_loss, 'cls_loss': cls_loss}

        return loss_dict

    def get_boxes(self, x, score_thresh=0.4):
        xs = self.feature_extractor(x)
        loc_preds, cls_preds = self.rpn(xs)
        scores = cls_preds[..., self.label_offset:].contiguous()
        anchors, _ = self.box_coder(xs)
        targets = self.box_coder.decode(anchors, loc_preds, scores, x.size(1), score_thresh=score_thresh)
        return targets

    @classmethod
    def tiny_rnn_fpn(cls, in_channels, num_classes, act='sigmoid'):
        return cls(FPN, BoxHead, in_channels, num_classes, act, ratios=[1.0], scales=[1.0, 1.5])

    @classmethod
    def mobilenet_v2_fpn(cls, in_channels, num_classes, act='sigmoid'):
        return cls(MobileNetFPN, BoxHead, in_channels, num_classes, act)

    @classmethod
    def resnet50_fpn(cls, in_channels, num_classes, act='sigmoid'):
        return cls(ResNet50FPN, BoxHead, in_channels, num_classes, act)

    @classmethod
    def resnet50_ssd(cls, in_channels, num_classes, act='softmax', loss='_ohem_loss'):
        backbone = lambda x: ResNet50FPN(x, no_fpn=True)
        return cls(backbone, SSDBoxHead, in_channels, num_classes, act, 
                    ratios=[1./3, 1./2, 1, 2, 3], scales=[1.0,1.5])