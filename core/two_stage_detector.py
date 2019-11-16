from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.ops.poolers as pool
from core.utils import box
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

        self.first_stage = rpn(self.feature_extractor.cout, self.box_coder.num_anchors, 1, 'sigmoid')

        feat_names = ['feat'+str(i) for i in range(self.feature_extractor.levels)]
        self.roi_pool = pool.MultiScaleRoIAlign(feat_names, 5, 2)
        self.second_stage = rpn(self.feature_extractor.cout,
                                         self.box_coder.num_anchors,
                                         self.num_classes + self.label_offset, act)


        self.criterion = DetectionLoss('sigmoid_focal_loss')


    def reset(self):
        self.feature_extractor.reset()

    def forward(self, x, score_thresh=0.4):
        xs = self.feature_extractor(x)
        loc_preds, cls_preds = self.first_stage(xs)
        scores = cls_preds.contiguous()
        anchors, anchors_xyxy = self.box_coder(xs)
        rois = self.box_coder.decode(anchors, loc_preds, scores, x.size(1), score_thresh=score_thresh)  
        image_sizes = [x.shape[-2:]]*x.size(0)*x.size(1)
        sources = {'feat'+str(i):item for i, item in enumerate(xs)}
        allboxes, sizes = self.gather_boxes(rois)
        if len(allboxes) > 0:
            out = self.roi_pool(sources, allboxes, image_sizes)
            loc_preds2, cls_preds2 = self.second_stage(out)
        else:
            loc_preds2, cls_preds2 = None, None
        
        out_dic = {
            'first_stage':{'loc':loc_preds, 'cls': cls_preds, 'boxes': rois, 'sizes': sizes},
            'second_stage':{'loc': loc_preds2, 'cls': cls_preds2},
            'anchors': (anchors, anchors_xyxy)
        }
        return out_dic

    def gather_boxes(self, rois):
        #this expects list of tensor of shape N, 4
        sizes = []
        allboxes = []
        for t in range(x.size(0)):
            for i in range(x.size(1)):
                boxes, _, _ = rois[t][i]
                num = len(boxes) if boxes is not None else 0
                sizes += [num]
                if num > 0:
                    boxes = box.change_box_order(boxes, 'xyxy2xywh')
                    allboxes += [boxes]
        return allboxes, sizes

    def compute_loss(self, x, targets):
        out_dic = self(x)
       
        with torch.no_grad():
            anchors, anchors_xyxy = out_dic['anchors']
            #binarize targets (whatever y => 1)
            loc_targets, cls_targets = self.box_coder.encode(anchors, anchors_xyxy, targets)

        #first stage
        loc_preds, cls_preds = out_dic['first_stage']['loc'], out_dic['first_stage']['cls']
        loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss_dict = {'loc1': loc_loss, 'cls1': cls_loss}

        #second stage
        loc_preds2, cls_preds2 = out_dic['second_stage']['loc'], out_dic['second_stage']['cls']
        if loc_preds is not None:


        return loss_dict

    """ def get_boxes(self, x, score_thresh=0.4):
        xs = self.feature_extractor(x)
        loc_preds, cls_preds = self.rpn(xs)
        scores = cls_preds[..., self.label_offset:].contiguous()
        targets = self.box_coder.decode(xs, loc_preds, scores, x.size(1), score_thresh=score_thresh)
        return targets  """

    

if __name__ == '__main__':
    x = torch.rand(1, 2, 3, 128, 128)

    net = TwoStageDetector(cin=3)

    y = net(x)
