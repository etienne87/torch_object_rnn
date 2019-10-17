from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


import torch
import torch.nn as nn
from core.utils import box, opts
import numpy as np


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

    def decode_boxes_from_anchors(self, anchors, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        xy = loc_preds[:,:2] * self.variances[0] * anchors[:,2:] + anchors[:,:2]
        wh = torch.exp(loc_preds[:,2:]* self.variances[1]) * anchors[:,2:]

        box_preds = torch.cat([xy-wh/2, xy+wh/2], 1)

        boxes = []
        labels = []
        scores = []
        num_classes = cls_preds.size(1)
        for i in range(num_classes-1):
            score = cls_preds[:,i+1]  # class i corresponds to (i+1) column
            mask = score > score_thresh
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

    def encode_boxes(self, features, targets):
        loc_targets, cls_targets = [], []
        device = features[0].device
        anchors = self(features) #1,A,4

        for i in range(len(targets)):
            boxes, labels = targets[i][:, :4], targets[i][:, -1]
            boxes, labels = boxes.to(device), labels.to(device) + self.label_offset
            loc_t, cls_t = self.encode_boxes_from_anchors(anchors, boxes, labels)
            loc_targets.append(loc_t.unsqueeze(0))
            cls_targets.append(cls_t.unsqueeze(0).long())

        loc_targets = torch.cat(loc_targets, dim=0)  # (N,#anchors,4)
        cls_targets = torch.cat(cls_targets, dim=0)  # (N,#anchors,C)

        return loc_targets, cls_targets


    def encode_boxes_from_anchors(self, anchors, gt_boxes, labels):
        anchors_xyxy = box.change_box_order(anchors, "xywh2xyxy")
        boxes, cls_targets = box.assign_priors(gt_boxes, labels, anchors_xyxy,
                                               self.fg_iou_threshold, self.bg_iou_threshold)
        boxes = box.change_box_order(boxes, "xyxy2xywh")
        loc_xy = (boxes[:, :2] - anchors[:, :2]) / anchors[:, 2:] / self.variances[0]
        loc_wh = torch.log(boxes[:, 2:] / anchors[:, 2:])  / self.variances[1]
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        return loc_targets, cls_targets

    def bbox_to_deltas(self, boxes, anchors):
        default_boxes = anchors[None]
        boxes = box.change_box_order(boxes, "xyxy2xywh")
        boxes[..., :2] = boxes[..., :2].clamp_(1, int(1e3))
        loc_xy = (boxes[..., :2] - default_boxes[..., :2]) / default_boxes[..., 2:] / self.variances[0]
        loc_wh = torch.log(boxes[..., 2:] / default_boxes[..., 2:]) / self.variances[1]
        deltas = torch.cat([loc_xy, loc_wh], 2)
        return deltas

    def encode_txn_boxes(self, features, targets):
        # Strategy: We pad box frames to enable batch optimization

        anchors, anchors_xyxy = self(features)
        device = features[0].device

        # START
        torch.cuda.synchronize()
        start = time.time()


        tbins, batchsize = len(targets), len(targets[0])
        total = tbins * batchsize
        max_size = max([max([len(frame) for frame in time]) for time in targets])
        max_size = max(2, max_size)
        gt_padded = torch.ones((tbins, batchsize, max_size, 5), dtype=torch.float32) * -1
        for t in range(len(targets)):
            for i in range(len(targets[t])):
                frame = targets[t][i]
                gt_padded[t, i, :len(frame)] = frame
                gt_padded[t, i, :len(frame), 4] += self.label_offset
        gt_padded = gt_padded.to(device)


        gt_boxes = gt_padded[..., :4]
        gt_labels = gt_padded[..., 4].long()

        # V1 Batched IOUS
        # ious = box.batch_box_iou(anchors_xyxy, gt_boxes.view(total, max_size, 4))
        # ious = ious.view(tbins, batchsize, len(anchors_xyxy), max_size)

        # V2 Batched IOUS (Simpler)
        ious = box.box_iou(anchors_xyxy, gt_boxes.view(-1, 4))
        ious = ious.reshape(len(anchors), total, max_size).permute(1, 0, 2).contiguous()
        ious = ious.view(tbins, batchsize, len(anchors_xyxy), max_size)

        batch_best_target_per_prior, batch_best_target_per_prior_index = ious.max(-1) # [T, N, A]
        _, batch_best_prior_per_target_index = ious.max(-2) # [T, N, M]

        for target_index in range(max_size):
            index = batch_best_prior_per_target_index[..., target_index:target_index+1]
            batch_best_target_per_prior_index.scatter_(2, index, target_index)
            batch_best_target_per_prior.scatter_(2, index, 2.0)


        mask_bg = batch_best_target_per_prior < self.bg_iou_threshold
        mask_ign = (batch_best_target_per_prior > self.bg_iou_threshold) * (batch_best_target_per_prior < self.fg_iou_threshold)
        labels = torch.gather(gt_labels, 2, batch_best_target_per_prior_index)
        labels[mask_ign] = -1
        labels[mask_bg] = 0
        dumdum = batch_best_target_per_prior_index[...,None].expand(tbins, batchsize, len(anchors), 4)
        boxes = torch.gather(gt_boxes, 2, dumdum)


        # memory = ious.shape[0] * ious.shape[1] * ious.shape[2] * 16 / (1024.0**3)
        # print('memory taken by iou matrix: ', memory)

        """
        loc_targets, cls_targets = [], []
        for t in range(len(targets)):
            for i in range(len(targets[t])):
                # num_boxes = min(1, len(targets[t][i]))
                boxes, labels = gt_boxes[t, i], gt_labels[t, i]


                best_target_per_prior, best_target_per_prior_index = batch_best_target_per_prior[t, i], batch_best_target_per_prior_index[t, i]
                best_prior_per_target_index = batch_best_prior_per_target_index[t, i]

                # faster, loop is done above
                loc_t, cls_t = box.assign_prior_with_best_overlaps_no_for_loop(best_target_per_prior, best_target_per_prior_index,
                                     boxes, labels, self.fg_iou_threshold, self.bg_iou_threshold)


                # a bit faster, loop is done inside
                # loc_t, cls_t = box.assign_prior_with_best_overlaps(best_target_per_prior, best_target_per_prior_index,
                #                      best_prior_per_target_index,
                #                      boxes, labels, self.fg_iou_threshold, self.bg_iou_threshold)

                # a bit faster (ious in parallel)
                # ious_ti = ious[t, i, :, :num_boxes]
                # loc_t, cls_t = box.assign_priors_with_iou(ious_ti, boxes, labels,
                #                                        self.fg_iou_threshold, self.bg_iou_threshold)

                # the most stupid (no parallel)
                # loc_t, cls_t = box.assign_priors(boxes, labels, anchors_xyxy,
                #                                          self.fg_iou_threshold, self.bg_iou_threshold)

                loc_targets.append(loc_t.unsqueeze(0))
                cls_targets.append(cls_t.unsqueeze(0))


        boxes = torch.cat(loc_targets, dim=0)  # (N,#anchors,4)
        labels = torch.cat(cls_targets, dim=0)  # (N,#anchors,C)
        """

        loc_targets = self.bbox_to_deltas(boxes, anchors)
        cls_targets = labels


        torch.cuda.synchronize()
        print('current encoding: ', time.time() - start)

        # diff = cls_targets - cls_targets_v2
        # print('DIFF: ', diff.abs().max().item())

        return loc_targets, cls_targets

    def encode_txn_boxes_old(self, features, targets):
        anchors, _ = self(features)
        device = features[0].device

        torch.cuda.synchronize()
        start = time.time()
        loc_targets, cls_targets = [], []
        for t in range(len(targets)):
            for i in range(len(targets[t])):
                if len(targets[t][i]) == 0:
                    targets[t][i] = torch.ones((1, 5), dtype=torch.float32) * -1

                boxes, labels = targets[t][i][:, :4], targets[t][i][:, -1]
                boxes, labels = boxes.to(device), labels.to(device)
                loc_t, cls_t = self.encode_boxes_from_anchors(anchors, boxes, labels)
                loc_targets.append(loc_t.unsqueeze(0))
                cls_targets.append(cls_t.unsqueeze(0).long())

        loc_targets = torch.cat(loc_targets, dim=0)  # (N,#anchors,4)
        cls_targets = torch.cat(cls_targets, dim=0)  # (N,#anchors,C)

        torch.cuda.synchronize()
        print('current encoding: ', time.time() - start)

        return loc_targets, cls_targets

    def decode_txn_boxes(self, features, loc_preds, cls_preds, batchsize, score_thresh):
        anchors = self(features)

        loc_preds = opts.batch_to_time(loc_preds, batchsize)
        cls_preds = opts.batch_to_time(cls_preds, batchsize)
        targets = []
        for t in range(loc_preds.size(0)):
            targets_t = []
            for i in range(loc_preds.size(1)):
                boxes, labels, scores = self.decode_boxes_from_anchors(anchors,    loc_preds[t, i].data,
                                                                                    cls_preds[t, i].data,
                                                                                    score_thresh=score_thresh,
                                                                                    nms_thresh=0.6)
                targets_t.append((boxes, labels, scores))
            targets.append(targets_t)
        return targets


if __name__ == '__main__':
    from core.ssd.box_coder import SSDBoxCoder, get_box_params_fixed_size
    # x = torch.randn(3, 128, 8, 8)

    # layer = AnchorLayer(32, 2**3, [1, 0.5, 2], [1])
    # anchors = layer(x)
    # print('anchor shape: ', anchors.shape)
    # print(anchors)
    #
    # anchors = layer(torch.randn(3, 128, 1, 1))
    # print(anchors.shape)
    # print(anchors)

    imsize = 512
    sources = []
    for level in [3,4,5,6]:
        sources.append(torch.rand(3, 16, imsize>>level, imsize>>level))

    box_coder = Anchors(ratios=[0.5, 1, 2], scales=[1, 2**(1./3), 2**(2./3)], base_size=24,
                        label_offset=1, fg_iou_threshold=0.7, bg_iou_threshold=0.4)


    class FakeSSD:
        def __init__(self, height=imsize, width=imsize):
            self.height, self.width = height, width
            self.fm_sizes, self.steps, self.box_sizes = get_box_params_fixed_size(sources, height, width)
            self.aspect_ratios = box_coder.ratios
            self.scales = box_coder.scales
            self.num_anchors = len(self.aspect_ratios) * len(self.scales)

    test = FakeSSD()
    box_coder2 = SSDBoxCoder(test, 0.7, 0.4)


    anchors = box_coder(sources)
    anchors_xyxy = box.change_box_order(anchors, "xywh2xyxy")


    diff = anchors - box_coder2.default_boxes
    diff2 = anchors_xyxy - box_coder2.default_boxes_xyxy

    print('diff default boxes: ', diff.abs().max().item(), diff.sum().item())
    print('diff default boxesxyxy: ', diff2.abs().max().item(), diff2.sum().item())

    #take some boxes & check if same default boxes come out
    from datasets.moving_box_detection import SquaresVideos

    dataset = SquaresVideos(t=10, c=1, h=256, w=256, batchsize=2, mode='diff', max_objects=20, render=False)
    _, targets, _ = dataset[0]
    #boxes = boxes[0]
    # print('boxes: ', boxes)

    # loc_t, cls_t = box_coder.encode_boxes(sources, boxes)
    #
    # print(loc_t.shape, cls_t.shape)
    #
    # loc_t2, cls_t2 = box_coder2.encode_boxes(boxes)

    boxes, labels = targets[0][:, :4], targets[0][:, -1]
    # loc_t, cls_t = box_coder.encode_boxes_from_anchors(anchors, boxes.clone(), labels.clone())

    import pdb
    pdb.set_trace()
    #
    # loc_t2, cls_t2 = box_coder2.encode(boxes, labels)
    print('box coder1: ', box_coder.fg_iou_threshold, box_coder.bg_iou_threshold)
    loc_t, cls_t = box.assign_priors(boxes, labels + 1, anchors_xyxy,
                                               box_coder.fg_iou_threshold, box_coder.bg_iou_threshold)

    print('box coder2 : ', box_coder2.fg_iou_threshold, box_coder2.bg_iou_threshold)
    loc_t2, cls_t2 = box.assign_priors(boxes, labels + 1, anchors_xyxy,
                                       box_coder2.fg_iou_threshold, box_coder2.bg_iou_threshold)

    diff_loc = loc_t - loc_t2
    diff_cls = cls_t - cls_t2

    # print(diff_cls[diff_cls != 0])
    #
    # import pdb
    # pdb.set_trace()
    #
    # print('diff loc encoding: ', diff_loc.abs().max())
    print('diff cls encoding: ', diff_cls.abs().max())