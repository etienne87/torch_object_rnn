from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from core.utils import box
import numpy as np


class AnchorLayer(nn.Module):
    def __init__(self, box_size=32, ratios=[1], scales=[1]):
        super(AnchorLayer, self).__init__()
        self.num_anchors = len(scales) * len(ratios)
        box_sizes = AnchorLayer.generate_anchors(box_size, ratios, scales)
        self.register_buffer("box_sizes", box_sizes.view(-1))
        self.anchors = None

    @staticmethod
    def generate_anchors(box_size, ratios, scales):
        anchors = box_size * np.tile(scales, (2, len(ratios))).T
        areas = anchors[:, 0] * anchors[:, 1]
        anchors[:, 0] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 1] = anchors[:, 0] * np.repeat(ratios, len(scales))
        return torch.from_numpy(anchors).float()

    @staticmethod
    def make_grid(height, width, num_anchors):
        grid_h, grid_w = torch.meshgrid([torch.linspace(0.5, height + 0.5, height),
                                         torch.linspace(0.5, width + 0.5, width)
                                         ])
        grid = torch.cat([grid_w[..., None], grid_h[..., None]], dim=-1)

        grid = grid[:,:,None,:].expand(height, width, num_anchors, 2)
        return grid

    def forward(self, x):
        height, width = x.shape[-2:]
        if self.anchors is None or self.anchors.shape[-2:] != (height, width) or self.anchors.device != x.device:
            grid = AnchorLayer.make_grid(height, width, self.num_anchors)
            wh = torch.zeros((self.num_anchors * 2, height, width), dtype=x.dtype, device=x.device) + self.box_sizes.view(self.num_anchors * 2, 1, 1)
            wh = wh.permute([1, 2, 0]).view(height, width, self.num_anchors, 2)
            self.anchors = torch.cat([grid, wh], dim=-1)
        return self.anchors.view(-1, 4)


class Anchors(nn.Module):
    def __init__(self, **kwargs):
        super(self, Anchors).__init__()

        self.pyramid_levels = kwargs.get("pyramid_levels", [3, 4, 5, 6])
        self.strides = kwargs.get("strides", [2 ** x for x in self.pyramid_levels])
        self.sizes = kwargs.get("sizes", [2 ** (x + 2) for x in self.pyramid_levels])
        self.ratios = kwargs.get("ratios", np.array([0.5, 1, 2]))
        self.scales = kwargs.get("scales", np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        self.learnable = kwargs.get("learnable", False)
        self.fg_iou_threshold = kwargs.get("fg_iou_threshold", 0.5)
        self.bg_iou_threshold = kwargs.get("bg_iou_threshold", 0.4)
        self.num_anchors = len(self.scales) * len(self.ratios)
        self.label_offset = kwargs.get("label_offset", 0) #0 by default, has to be 1 if using softmax

        self.anchor_generators = nn.ModuleList()
        for box_size, in_channels in zip(self.sizes, self.in_channels):
            self.anchor_generators.append(AnchorLayer(box_size, self.ratios, self.scales))

    def forward(self, features):
        default_boxes = []
        for feature_map, anchor_layer in zip(features, self.anchor_generators):
            anchors = anchor_layer(feature_map)
            default_boxes.append(anchors)
        return torch.cat(default_boxes, dim=0)

    def encode_boxes_from_anchors(self, anchors, gt_boxes, labels):
        anchors_xyxy = box.change_box_order(anchors, "xywh2xyxy")
        boxes, cls_targets = box.assign_priors(gt_boxes, labels + self.label_offset, anchors_xyxy,
                                               self.fg_iou_threshold, self.bg_iou_threshold)
        boxes = box.change_box_order(boxes, 'xyxy2xywh')
        loc_xy = (boxes[:, :2] - anchors[:, :2]) / anchors[:, 2:]
        loc_wh = torch.log(boxes[:, 2:] / anchors[:, 2:])
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        return loc_targets, cls_targets

    def encode_boxes_from_features(self, features, targets):
        loc_targets, cls_targets = [], []
        device = features.device
        anchors = self(features) #1,A,4

        for i in range(len(targets)):
            boxes, labels = targets[i][:, :4], targets[i][:, -1]
            boxes, labels = boxes.to(device), labels.to(device)
            loc_t, cls_t = self.encode_boxes_from_anchors(anchors, boxes, labels)
            loc_targets.append(loc_t.unsqueeze(0))
            cls_targets.append(cls_t.unsqueeze(0).long())

        loc_targets = torch.cat(loc_targets, dim=0)  # (N,#anchors,4)
        cls_targets = torch.cat(cls_targets, dim=0)  # (N,#anchors,C)

        return loc_targets, cls_targets

    def encode_boxes_from_features_txn(self, features, targets):
        loc_targets, cls_targets = [], []
        device = features.device
        anchors = self(features)

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

        return loc_targets, cls_targets


if __name__ == '__main__':
    x = torch.randn(3, 128, 16, 16)

    layer = AnchorLayer(32, [1, 1./2, 2], [1, 2**1./3, 2**1./6])
    anchors = layer(x)
    print(anchors.shape)
    print(anchors)

    anchors = layer(torch.randn(3, 128, 1, 1))
    print(anchors.shape)
    print(anchors)

    # layer = Anchors()



