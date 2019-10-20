from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


import torch
import torch.nn as nn
from core.utils import box, opts
import numpy as np
from torchvision.ops.boxes import batched_nms


class AnchorLayer(nn.Module):
    def __init__(self, box_size=32, stride=8, ratios=[1], scales=[1]):
        super(AnchorLayer, self).__init__()
        self.num_anchors = len(scales) * len(ratios)
        box_sizes = AnchorLayer.generate_anchors(box_size, ratios, scales)
        self.register_buffer("box_sizes", box_sizes.view(-1))
        self.anchors = None
        self.stride = stride

    @staticmethod
    def generate_anchors(box_size, ratios, scales):
        anchors = box_size * np.tile(scales, (2, len(ratios))).T
        areas = anchors[:, 0] * anchors[:, 1]
        anchors[:, 0] = np.sqrt(areas * np.repeat(ratios, len(scales)))
        anchors[:, 1] = anchors[:, 0] / np.repeat(ratios, len(scales))
        return torch.from_numpy(anchors).float()

    def make_grid(self, height, width):

        grid_h, grid_w = torch.meshgrid([torch.linspace(0.5 * self.stride, (height-1 + 0.5) * self.stride, height),
                                         torch.linspace(0.5 * self.stride, (width-1 + 0.5) * self.stride, width)
                                         ])
        grid = torch.cat([grid_w[..., None], grid_h[..., None]], dim=-1)

        grid = grid[:,:,None,:].expand(height, width, self.num_anchors, 2)
        return grid

    def forward(self, x):
        height, width = x.shape[-2:]
        if self.anchors is None or self.anchors.shape[-2:] != (height, width) or self.anchors.device != x.device:
            grid = self.make_grid(height, width).to(x.device)
            wh = torch.zeros((self.num_anchors * 2, height, width), dtype=x.dtype, device=x.device) + self.box_sizes.view(self.num_anchors * 2, 1, 1)
            wh = wh.permute([1, 2, 0]).view(height, width, self.num_anchors, 2)
            self.anchors = torch.cat([grid, wh], dim=-1)

        return self.anchors.view(-1, 4)


class Anchors(nn.Module):
    def __init__(self, **kwargs):
        super(Anchors, self).__init__()
        self.pyramid_levels = kwargs.get("pyramid_levels", [3, 4, 5, 6])
        self.strides = kwargs.get("strides", [2 ** x for x in self.pyramid_levels])
        self.base_size = kwargs.get("base_size", 32)
        self.sizes = kwargs.get("sizes", [self.base_size * 2 ** x for x in range(len(self.pyramid_levels))])
        self.ratios = kwargs.get("ratios", np.array([0.5, 1, 2]))
        self.scales = kwargs.get("scales", np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        self.fg_iou_threshold = kwargs.get("fg_iou_threshold", 0.5)
        self.bg_iou_threshold = kwargs.get("bg_iou_threshold", 0.4)
        self.num_anchors = len(self.scales) * len(self.ratios)
        self.label_offset = kwargs.get("label_offset", 0) #0 by default, has to be 1 if using softmax

        self.variances = (0.1, 0.2)
        self.nms_type = kwargs.get("nms_type", "soft_nms")
        self.nms = box.box_soft_nms if self.nms_type == "soft_nms" else box.box_nms

        self.anchor_generators = nn.ModuleList()
        for i, (box_size, stride) in enumerate(zip(self.sizes, self.strides)):
            self.anchor_generators.append(AnchorLayer(box_size, stride, self.ratios, self.scales))
        self.anchors = None
        self.anchors_xyxy = None

    def forward(self, features):
        size = sum([(item.shape[-2]*item.shape[-1]*self.num_anchors) for item in features])
        if self.anchors is None or len(self.anchors) != size:
            default_boxes = []
            for feature_map, anchor_layer in zip(features, self.anchor_generators):
                anchors = anchor_layer(feature_map)
                default_boxes.append(anchors)
            self.anchors = torch.cat(default_boxes, dim=0)
            self.anchors_xyxy = box.change_box_order(self.anchors, "xywh2xyxy")

        return self.anchors, self.anchors_xyxy


    def encode(self, features, targets):
        # Strategy: We pad box frames to enable gpu optimization (from 300 ms -> 1 ms)

        anchors, anchors_xyxy = self(features)
        device = features[0].device

        if isinstance(targets[0], list):
            gt_padded = box.pack_boxes_list_of_list(targets, self.label_offset).to(device)
        else:
            gt_padded = box.pack_boxes_list(targets, self.label_offset).to(device)

        total = len(gt_padded)
        max_size = gt_padded.shape[1]
        gt_boxes = gt_padded[..., :4]
        gt_labels = gt_padded[..., 4].long()


        ious = box.batch_box_iou(anchors_xyxy, gt_boxes) # [N, A, M]

        batch_best_target_per_prior, batch_best_target_per_prior_index = ious.max(-1) # [N, A]
        _, batch_best_prior_per_target_index = ious.max(-2) # [N, M]

        #V1 (fast, but no guarantee of uniqueness for low quality matches)
        for target_index in range(max_size):
            index = batch_best_prior_per_target_index[..., target_index:target_index+1]
            batch_best_target_per_prior_index.scatter_(-1, index, target_index)
            batch_best_target_per_prior.scatter_(-1, index, 2.0)

        mask_bg = batch_best_target_per_prior < self.bg_iou_threshold
        mask_ign = (batch_best_target_per_prior > self.bg_iou_threshold) * (batch_best_target_per_prior < self.fg_iou_threshold)
        labels = torch.gather(gt_labels, 1, batch_best_target_per_prior_index)
        labels[mask_ign] = -1
        labels[mask_bg] = 0
        index = batch_best_target_per_prior_index[...,None].expand(total, len(anchors), 4)
        boxes = torch.gather(gt_boxes, 1, index)


        loc_targets = box.bbox_to_deltas(boxes, anchors[None])#.view(-1, len(anchors), 4)
        cls_targets = labels.view(-1, len(anchors))


        return loc_targets, cls_targets


    def decode(self, features, loc_preds, cls_preds, batchsize, score_thresh, nms_thresh=0.6):
        torch.cuda.synchronize()
        start = time.time()
        anchors, _ = self(features)
        box_preds = box.deltas_to_bbox(loc_preds, anchors)

        # batched
        boxes, labels, scores, batch_index = self.batch_decode(box_preds, cls_preds, score_thresh, nms_thresh)
        tbins = len(cls_preds) // batchsize
        targets = [[(None,None,None) for i in range(batchsize)] for t in range(tbins)]
        for t in range(tbins):
            for i in range(batchsize):
                index = t*batchsize + i
                mask = batch_index == index
                boxe = boxes[mask]
                score = scores[mask]
                label = labels[mask]
                if len(boxe):
                    targets[t][i] = (boxe, label, score)

        # sequential, less batched (if call decode_boxes_from_anchors this is seriously slow)
        # box_preds = opts.batch_to_time(box_preds, batchsize)
        # cls_preds = opts.batch_to_time(cls_preds, batchsize)
        #
        # targets = []
        # for t in range(box_preds.size(0)):
        #     targets_t = []
        #     for i in range(box_preds.size(1)):
        #         boxes, labels, scores = self.batch_decode_boxes_from_anchors(box_preds[t, i].data,
        #                                                                 cls_preds[t, i].data,
        #                                                                 score_thresh=score_thresh,
        #                                                                 nms_thresh=nms_thresh)
        #         targets_t.append((boxes, labels, scores))
        #     targets.append(targets_t)

        torch.cuda.synchronize()
        print(time.time()-start, ' s decoding')
        return targets

    def decode_boxes_from_anchors(self, box_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        num_classes = cls_preds.shape[1] - self.label_offset

        maskall = cls_preds > score_thresh
        boxes = []
        labels = []
        scores = []
        for i in range(num_classes):
            score = cls_preds[:,i+self.label_offset]
            mask = maskall[:,i+self.label_offset] #score > score_thresh
            if not mask.any():
                continue


            box = box_preds[mask]
            score = score[mask]

            keep = self.nms(box, score, nms_thresh)
            boxes.append(box[keep])
            labels.append(torch.LongTensor(len(box[keep])).fill_(i))
            scores.append(score[keep])

        if len(boxes) > 0:
            boxes = torch.cat(boxes, 0)
            labels = torch.cat(labels, 0)
            scores = torch.cat(scores, 0)
            return boxes, labels, scores
        else:
            return None, None, None

    def batch_decode_boxes_from_anchors(self, box_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        num_classes = cls_preds.shape[1] - self.label_offset
        num_anchors = len(box_preds)
        boxes = box_preds.unsqueeze(0).expand(num_classes, num_anchors, 4).contiguous() #C,A,4
        scores = cls_preds[..., self.label_offset:].transpose(1,0).contiguous() #C,A

        boxes = boxes.view(-1, 4)
        scores = scores.view(-1)

        idxs = torch.arange(num_classes, dtype=torch.long)[None,:].to(boxes)
        idxs = idxs.expand(num_anchors, num_classes).transpose(1,0).contiguous()
        idxs = idxs.view(-1)

        mask = scores >= score_thresh
        boxesf = boxes[mask].contiguous()
        scoresf = scores[mask].contiguous()
        idxsf = idxs[mask].contiguous()

        keep = batched_nms(boxesf, scoresf, idxsf, nms_thresh)

        if len(keep) == 0:
            return None, None, None

        boxes = boxesf[keep]
        scores = scoresf[keep]
        labels = idxsf[keep]%num_classes
        return boxes, labels, scores

    def batch_decode(self, box_preds, cls_preds, score_thresh, nms_thresh):

        num_classes = cls_preds.shape[-1] - self.label_offset
        num_anchors = box_preds.shape[1]
        boxes = box_preds.unsqueeze(2).expand(-1, num_anchors, num_classes, 4).contiguous()

        scores = cls_preds[..., self.label_offset:].contiguous()
        boxes = boxes.view(-1, 4)
        scores = scores.view(-1)
        rows = torch.arange(len(box_preds), dtype=torch.long)[:, None]
        cols = torch.arange(num_classes, dtype=torch.long)[None, :]
        idxs = rows * num_classes + cols
        idxs = idxs.unsqueeze(1).expand(len(box_preds), num_anchors, num_classes)
        idxs = idxs.to(scores).view(-1)

        mask = scores >= score_thresh
        boxesf = boxes[mask].contiguous()
        scoresf = scores[mask].contiguous()
        idxsf = idxs[mask].contiguous()

        keep = batched_nms(boxesf, scoresf, idxsf, nms_thresh)

        if len(keep) == 0:
            return None, None, None

        boxes = boxesf[keep]
        scores = scoresf[keep]
        labels = idxsf[keep] % num_classes
        batch_index = idxsf[keep] // num_classes
        return boxes, labels, scores, batch_index.cpu()