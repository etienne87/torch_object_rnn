from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from core.ssd.model import ConvRNNFeatureExtractor
from torchvision.models.detection import rpn



class WrapRPN(nn.Module):
    def __init__(self, num_classes, in_channels, height, width,
                                    fg_iou_thresh=0.7, bg_iou_thresh=0.3,
                                    batch_size_per_image=10, positive_fraction=1.0,
                                    pre_nms_top_n=1000, post_nms_top_n=100, nms_thresh=0.7):
        super(WrapRPN, self).__init__()

        self._backbone = ConvRNNFeatureExtractor(in_channels)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.height, self.width = height, width
        x = torch.randn(1, 1, self.in_channels, self.height, self.width)
        sources = self._backbone(x)
        sizes = [min(item.shape[-2:]) for item in sources]

        self._rpn_head = rpn.RPNHead(self._backbone.end_point_channels[0], 3)
        self._anchor_generator = rpn.AnchorGenerator(sizes=sizes,
                                                            aspect_ratios=(0.5, 1.0, 2.0))

        self._rpn = rpn.RegionProposalNetwork(self._anchor_generator,
                                                    self._rpn_head,
                                                     fg_iou_thresh, bg_iou_thresh,
                                                     batch_size_per_image, positive_fraction,
                                                     pre_nms_top_n, post_nms_top_n, nms_thresh)


    def reset(self):
        self._backbone.reset()

    def forward(self, x):
        features = self._backbone(x)
        return self._rpn.forward(self, x, features)[0]

    def compute_loss(self, x, targets):
        features = self._backbone(x)
        return self._rpn.forward(self, x, features, targets=None)[1]

