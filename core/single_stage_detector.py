from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


from core.losses import DetectionLoss
from core.backbones import Vanilla, FPN, MobileNetFPN, ResNet50FPN, ResNet50SSD
from core.anchors import Anchors
from core.rpn import BoxHead, SSDHead



class SingleStageDetector(nn.Module):
    def __init__(self, feature_extractor=FPN, rpn=BoxHead,
                 in_channels=3, 
                 num_classes=2,  
                 act='sigmoid', 
                 ratios=[0.5,1.0,2.0], 
                 scales=[1.0,2**1./3,2**2./3], 
                 nlayers=0,
                 loss='_focal_loss'):
        super(SingleStageDetector, self).__init__()
        self.label_offset = 1 * (act=='softmax')
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.feature_extractor = feature_extractor(in_channels)

        self.box_coder = Anchors(num_levels=self.feature_extractor.levels,
                                 scales=scales,
                                 ratios=ratios,
                                 allow_low_quality_matches=False,
                                 variances=[1.0,1.0],
                                 fg_iou_threshold=0.5, bg_iou_threshold=0.4)

        self.num_anchors = self.box_coder.num_anchors
        self.act = act

        if rpn == BoxHead:
            self.rpn = BoxHead(self.feature_extractor.cout, self.box_coder.num_anchors, self.num_classes + self.label_offset, act, nlayers)
        elif rpn == SSDHead:
            self.rpn = SSDHead(self.feature_extractor.out_channel_list, self.box_coder.num_anchors, self.num_classes + self.label_offset, act)
        else:
            raise NotImplementedError()

        self.criterion = DetectionLoss(act + loss) 

    def reset(self, mask=None):
        self.feature_extractor.reset(mask)

    def forward(self, x):
        xs = self.feature_extractor(x)
        return self.rpn(xs)

    def compute_loss(self, x, targets):
        xs = self.feature_extractor(x)
        loc_preds, cls_preds = self.rpn(xs)

        with torch.no_grad():
            anchors, anchors_xyxy = self.box_coder(xs, x.shape[-2:])
            loc_targets, cls_targets = self.box_coder.encode(anchors, anchors_xyxy, targets)

        assert cls_targets.shape[1] == cls_preds.shape[1]
        loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss_dict = {'loc': loc_loss, 'cls_loss': cls_loss}

        return loss_dict

    def get_boxes(self, x, score_thresh=0.4):
        xs = self.feature_extractor(x)
        loc_preds, cls_preds = self.rpn(xs)
        cls_preds = self.rpn.probas(cls_preds)
        scores = cls_preds[..., self.label_offset:].contiguous()
        anchors, _ = self.box_coder(xs, x.shape[-2:])
        targets = self.box_coder.decode(anchors, loc_preds, scores, x.size(1), score_thresh=score_thresh)
        return targets

    @classmethod
    def mnist_vanilla_rnn(cls, in_channels, num_classes, act='softmax', loss='_ohem_loss'):
        return cls(Vanilla, BoxHead, in_channels, num_classes, act, ratios=[1.0], scales=[1.0, 1.5], loss=loss)

    @classmethod
    def mnist_unet_rnn(cls, in_channels, num_classes, act='sigmoid', loss='_focal_loss'):
        return cls(FPN, BoxHead, in_channels, num_classes, act, ratios=[1.0], scales=[1.0, 1.5], loss=loss)

    @classmethod
    def mobilenet_v2_fpn(cls, in_channels, num_classes, act='sigmoid', loss='_focal_loss', nlayers=3):
        return cls(MobileNetFPN, BoxHead, in_channels, num_classes, act, loss=loss, nlayers=nlayers)

    @classmethod
    def resnet50_fpn(cls, in_channels, num_classes, act='sigmoid', loss='_focal_loss', nlayers=3):
        return cls(ResNet50FPN, BoxHead, in_channels, num_classes, act, loss=loss, nlayers=nlayers)

    @classmethod
    def resnet50_ssd(cls, in_channels, num_classes, act='sigmoid', loss='_focal_loss', nlayers=0):
        return cls(ResNet50SSD, SSDHead, in_channels, num_classes, act=act, loss=loss, nlayers=nlayers)
