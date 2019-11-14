'''
This contains the class to encode/ decode the gt into 'anchor-boxes'
This runs in parallel over all images at once by padding the gt with dummies
The dummies are however not encoded. This is enforced by the pytest.

Examples::

    >> sources = pyramid_network(x)
    >> anchors = anchors(sources)
    >> loc_targets, cls_targets = box_coder.encode(anchors, targets)

The module can resize its grid internally (so batches can change sizes)
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from core.utils import box, opts
import numpy as np
from torchvision.ops.boxes import batched_nms


class AnchorLayer(nn.Module):
    '''
    For one level of the pyramid: Manages One Grid (x,y,w,h)
    The anchors grid is (height, width, num_anchors_per_position, 4)
    The grid is cached, but changes if featuremap size changes
    '''
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
        if self.anchors is None or self.anchors.shape[0] != height or self.anchors.shape[1] != width or self.anchors.device != x.device:
            grid = self.make_grid(height, width).to(x.device)
            wh = torch.zeros((self.num_anchors * 2, height, width), dtype=x.dtype, device=x.device) + self.box_sizes.view(self.num_anchors * 2, 1, 1)
            wh = wh.permute([1, 2, 0]).view(height, width, self.num_anchors, 2)
            self.anchors = torch.cat([grid, wh], dim=-1)

        return self.anchors.view(-1, 4)


class Anchors(nn.Module):
    '''
    Pyramid of Anchoring Grids.
    Handle encoding/ decoding algorithms.
    Encoding uses padding in order to parallelize iou & assignement computation.
    Decoding uses "batched_nms" of torchvision to parallelize accross images and classes.
    '''
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
        self.allow_low_quality_matches = kwargs.get("allow_low_quality_matches", False)
        self.variances = (0.1, 0.2)
        self.anchor_generators = nn.ModuleList()
        for i, (box_size, stride) in enumerate(zip(self.sizes, self.strides)):
            self.anchor_generators.append(AnchorLayer(box_size, stride, self.ratios, self.scales))
        self.anchors = None
        self.anchors_xyxy = None
        self.idxs = None
        self.last_shapes = []
        self.set_low_quality_matches = self.set_low_quality_matches_v1

    def forward(self, features):
        shapes = [item.shape for item in features]
        if self.anchors is None or shapes != self.last_shapes:
            default_boxes = []
            for feature_map, anchor_layer in zip(features, self.anchor_generators):
                anchors = anchor_layer(feature_map)
                default_boxes.append(anchors)
            self.anchors = torch.cat(default_boxes, dim=0)
            self.anchors_xyxy = box.change_box_order(self.anchors, "xywh2xyxy")
        self.last_shapes = shapes
        return self.anchors, self.anchors_xyxy

    # @opts.cuda_time
    def encode(self, anchors, anchors_xyxy, targets):
        device = anchors.device

        if isinstance(targets[0], list):
            gt_padded, sizes = box.pack_boxes_list_of_list(targets)
        else:
            gt_padded, sizes = box.pack_boxes_list(targets)

        total = len(gt_padded)
        gt_padded = gt_padded.to(device)
        gt_boxes = gt_padded[..., :4]
        gt_labels = gt_padded[..., 4].long()

        ious = box.batch_box_iou(anchors_xyxy, gt_boxes) # [N, A, M]

        #make sure to not select the dummies
        mask = (gt_labels == -2).float().unsqueeze(1)
        ious = (-1 * mask) + ious * (1-mask)

        batch_best_target_per_prior, batch_best_target_per_prior_index = ious.max(-1) # [N, A]

        if self.allow_low_quality_matches:
            self.set_low_quality_matches(ious, batch_best_target_per_prior_index, batch_best_target_per_prior, sizes)

        mask_bg = batch_best_target_per_prior < self.bg_iou_threshold
        mask_ign = (batch_best_target_per_prior > self.bg_iou_threshold) * (batch_best_target_per_prior < self.fg_iou_threshold)
        labels = torch.gather(gt_labels, 1, batch_best_target_per_prior_index)
        labels[mask_ign] = -1
        labels[mask_bg] = 0
        index = batch_best_target_per_prior_index[...,None].expand(total, len(anchors), 4)
        boxes = torch.gather(gt_boxes, 1, index)

        loc_targets = box.bbox_to_deltas(boxes, anchors[None])
        cls_targets = labels.view(-1, len(anchors))

        return loc_targets, cls_targets

    def set_low_quality_matches_v1(self, ious, batch_best_target_per_prior_index, batch_best_target_per_prior, sizes):
        _, batch_best_prior_per_target_index = ious.max(-2)  # [N, M]
        for t in range(len(sizes)):
            max_size = sizes[t]
            best_prior_per_target_index = batch_best_prior_per_target_index[t, :max_size]
            for target_index, prior_index in enumerate(best_prior_per_target_index):
                batch_best_target_per_prior_index[t, prior_index] = target_index
                batch_best_target_per_prior[t, prior_index] = 2.0

    def set_low_quality_matches_v2(self, ious, matches, match_vals, sizes):
        highest_quality_foreach_gt, _ = ious.max(-2)  # [N, M]
        gt_pred_pairs_of_highest_quality = torch.nonzero(ious == highest_quality_foreach_gt.unsqueeze(1))
        batch_index = gt_pred_pairs_of_highest_quality[:, 0]
        pred_index = gt_pred_pairs_of_highest_quality[:, 1]
        gt_index = gt_pred_pairs_of_highest_quality[:, 2]
        matches[batch_index, pred_index] = gt_index
        match_vals[batch_index, pred_index] = 2.0

    @opts.cuda_time
    def decode(self, anchors, loc_preds, cls_preds, batchsize, score_thresh, nms_thresh=0.6):
        # loc_preds [N, C] (do not include background column)
        box_preds = box.deltas_to_bbox(loc_preds, anchors)

        # batch decoding
        num_classes = cls_preds.shape[-1]
        num_anchors = box_preds.shape[1]

        # Per-Column Decoding
        boxes = box_preds.unsqueeze(2).expand(-1, num_anchors, num_classes, 4).contiguous()
        scores = cls_preds
        boxes, scores, idxs = self.decode_per_image(boxes, scores,
                                                    num_anchors, num_classes,
                                                    len(box_preds), score_thresh, nms_thresh)
        labels = idxs % num_classes
        batch_index = idxs // num_classes

        tbins = len(cls_preds) // batchsize
        targets = [[(None,None,None) for _ in range(batchsize)] for _ in range(tbins)]

        bidx, sidx = batch_index.sort()
        bidx_vals, sizes = torch.unique(bidx.long(), return_counts=True)
        sidx_list = sidx.split(sizes.cpu().numpy().tolist())

        for bidx_val, group in zip(bidx_vals.cpu().numpy().tolist(), sidx_list):
            t = bidx_val // batchsize
            i = bidx_val % batchsize
            targets[t][i] = (boxes[group], labels[group], scores[group])

        return targets

    def batched_decode(self, boxes, scores, num_anchors, num_classes, batchsize, score_thresh, nms_thresh):
        rows = torch.arange(batchsize, dtype=torch.long)[:, None]
        cols = torch.arange(num_classes, dtype=torch.long)[None, :]
        idxs = rows * num_classes + cols
        idxs = idxs.unsqueeze(1).expand(batchsize, num_anchors, num_classes).contiguous()
        idxs = idxs.to(scores.device)
        boxes = boxes.view(-1, 4)
        scores = scores.view(-1)
        idxs = idxs.view(-1)
        mask = scores >= score_thresh
        boxesf = boxes[mask].contiguous()
        scoresf = scores[mask].contiguous()
        idxsf = idxs[mask].contiguous()
        topk = int(1e5)
        if len(boxesf) > topk:
            scoresf, idx = torch.sort(scoresf, descending=True)
            scoresf = scoresf[:topk]
            boxesf = boxesf[idx][:topk]
            idxsf = idxsf[idx][:topk]
        keep = batched_nms(boxesf, scoresf, idxsf, nms_thresh)
        boxes = boxesf[keep]
        scores = scoresf[keep]
        labels = idxsf[keep] % num_classes
        batch_index = idxsf[keep] // num_classes
        return boxes, scores, labels, batch_index

    def decode_per_image(self, boxes, scores, num_anchors, num_classes, batchsize, score_thresh, nms_thresh):
        rows = torch.arange(batchsize, dtype=torch.long)[:, None]
        cols = torch.arange(num_classes, dtype=torch.long)[None, :]
        idxs = rows * num_classes + cols
        idxs = idxs.unsqueeze(1).expand(batchsize, num_anchors, num_classes).contiguous()
        idxs = idxs.to(scores.device)
        labels = cols.expand(num_anchors, num_classes).to(scores.device).view(-1)
        allboxes = []
        allscores = []
        allidxs = []
        for i in range(batchsize):
            boxesi = boxes[i].view(-1, 4)
            scoresi = scores[i].view(-1)
            idxsi = idxs[i].view(-1)
            mask = scoresi >= score_thresh
            boxesf = boxesi[mask].contiguous()
            scoresf = scoresi[mask].contiguous()
            idxsf = idxsi[mask].contiguous()
            labelsf = labels[mask]
            keep = batched_nms(boxesf, scoresf, labelsf, nms_thresh)
            allboxes.append(boxesf[keep])
            allscores.append(scoresf[keep])
            allidxs.append(idxsf[keep])
        boxes = torch.cat(allboxes)
        scores = torch.cat(allscores)
        idxs = torch.cat(allidxs)
        return boxes, scores, idxs
