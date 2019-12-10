from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.ops.poolers as pool
import copy
from core.utils import box
from core.losses import DetectionLoss
from core.backbones import FPN
from core.anchors import Anchors
from core.rpn import BoxHead, FCHead

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def binarize_targets(targets):
    targets2 = copy.deepcopy(targets)
    for i in range(len(targets2)):
        for j in range(len(targets2[i])):
            targets2[i][j][:, 4] = 1
    return targets2

# def encode_rois(rois, targets):
    # rois : [B, K_max, 4]
    # targets: [B, M_max, 4]
    # compute iou : [B, K_max, M_max]
    # compute assignements...
    # algorithm should be similar to anchors 
    

class TwoStageDetector(nn.Module):
    def __init__(self, feature_extractor=FPN, rpn=BoxHead,
                 num_classes=2, cin=2, act='sigmoid',
                 ratios=[1.0], 
                 scales=[1.0,1.5]):
        super(TwoStageDetector, self).__init__()
        self.label_offset = 1 * (act=='softmax')
        self.num_classes = num_classes
        self.cin = cin

        self.feature_extractor = feature_extractor(cin)

        self.box_coder = Anchors(pyramid_levels=[i for i in range(3,3+self.feature_extractor.levels)],
                                 scales=scales,
                                 ratios=ratios,
                                 fg_iou_threshold=0.5, bg_iou_threshold=0.4)

        self.num_anchors = self.box_coder.num_anchors
        self.act = act

        self.first_stage = rpn(self.feature_extractor.cout, self.box_coder.num_anchors, 1, 'sigmoid', n_layers=0)

        feat_names = ['feat'+str(i) for i in range(self.feature_extractor.levels)]
        self.roi_pool = pool.MultiScaleRoIAlign(feat_names, 5, 2)
        self.second_stage = FCHead(self.feature_extractor.cout * 5 * 5, self.num_classes + self.label_offset, act)
        self.criterion = DetectionLoss('sigmoid_focal_loss')

    def reset(self):
        self.feature_extractor.reset()

    def forward(self, x, score_thresh=0.4):
        batchsize = x.size(1)
        xs = self.feature_extractor(x)
        loc_preds, cls_preds = self.first_stage(xs)
        anchors, anchors_xyxy = self.box_coder(xs)
        proposals = self.box_coder.decode(anchors, loc_preds, cls_preds.sigmoid(), batchsize, score_thresh=score_thresh)  
        image_sizes = [x.shape[-2:]]*x.size(0)*x.size(1)
        sources = {'feat'+str(i):item for i, item in enumerate(xs)}
        rois, rois_xyxy, sizes, batch_index = self.gather_boxes(proposals)
        if len(rois) > 0:
            out = self.roi_pool(sources, rois, image_sizes)
            rois, rois_xyxy = torch.cat(rois), torch.cat(rois_xyxy)   
            loc_preds2, cls_preds2 = self.second_stage(out)
        else:
            loc_preds2, cls_preds2, rois, rois_xyxy, batch_index = None, None, None, None, None
        
        out_dic = Struct(**{
            'first_stage': Struct(**{'loc':loc_preds, 'cls': cls_preds, 
            'proposals': proposals, 'rois': rois, 'rois_xyxy': rois_xyxy, 'sizes': sizes, 'idxs': batch_index}),
            'second_stage':Struct(**{'loc': loc_preds2, 'cls': cls_preds2}),
            'anchors': anchors,
            'anchors_xyxy': anchors_xyxy
        })
        return out_dic

    def gather_boxes(self, proposals):
        #this expects list of tensor of shape N, 4
        idxs = []
        sizes = []
        rois = []
        rois_xyxy = []
        stride = len(proposals)
        for t in range(len(proposals)):
            for i in range(len(proposals[t])):
                boxes, _, _ = proposals[t][i]
                num = len(boxes) if boxes is not None else 0
                sizes += [num]

                if num > 0:
                    boxes = boxes.detach()
                    rois_xyxy += [boxes]
                    rois += [box.change_box_order(boxes, 'xyxy2xywh')] 
                    idxs += [t*stride + i] * num  

        idxs = torch.LongTensor(idxs).to(rois[0].device)
        return rois, rois_xyxy, sizes, idxs

    def compute_loss(self, x, targets):
        out = self(x)
       
        #first stage loss
        with torch.no_grad():
            loc_targets, cls_targets = self.box_coder.encode(out.anchors, out.anchors_xyxy, binarize_targets(targets))
        loc_loss, cls_loss = self.criterion(out.first_stage.loc, loc_targets, out.first_stage.cls, cls_targets)
        loss_dict = {'loc1': loc_loss, 'cls1': cls_loss}  
        
        #second stage loss
        if out.second_stage.loc is not None:    
            with torch.no_grad():
                loc_targets2, cls_targets2 = self.box_coder.encode(out.first_stage.rois, out.first_stage.rois_xyxy, targets)  
            import pdb;pdb.set_trace()
            loc_loss, cls_loss = self.criterion(out.second_stage.loc, loc_targets2, out.second_stage.cls, cls_targets2)
            loss_dict.update({'loc2': loc_loss, 'cls2': cls_loss})

        return loss_dict

    def get_boxes(self, x, score_thresh=0.4):
        batchsize = x.size(1)
        out = self(x)
        
        scores, idxs = out.second_stage.cls.sigmoid().max(dim=1)
        idxs = out.first_stage.batch_index * self.num_classes + idxs 
        box_preds = box.deltas_to_bbox(out.second_stage.loc, out.first_stage.rois, [1,1])
        boxes, scores, labels, batch_index = self.box_coder.batched_decode_with_idxs(box_preds, scores, idxs, self.num_anchors, self.num_classes, batchsize, 0.5, 0.5) 
        targets = self.box_coder.flatten_box_list_to_list_of_list(boxes, scores, labels, batch_index, batchsize)
    
        return targets 

    

if __name__ == '__main__':
    x = torch.rand(1, 2, 3, 128, 128)

    net = TwoStageDetector(cin=3)

    y = net(x)
