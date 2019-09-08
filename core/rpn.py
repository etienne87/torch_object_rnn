from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from core.ssd.model import Trident
from torchvision.models.detection import rpn
from torchvision.models.detection.image_list import ImageList



class WrapRPN(nn.Module):
    def __init__(self, num_classes, in_channels, height, width,
                                    fg_iou_thresh=0.7, bg_iou_thresh=0.3,
                                    batch_size_per_image=10, positive_fraction=1.0,
                                    nms_thresh=0.7):
        super(WrapRPN, self).__init__()


        pre_nms_top_n = {'training':10000, 'testing':1000}
        post_nms_top_n = {'training':100, 'testing':10}


        self._backbone = Trident(in_channels)

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

    def compute_loss(self, x, y):
        features = self._backbone(x)
        device = x.device
        targets_ = []
        for t in range(len(y)):
            for i in range(len(y[t])):
                boxes, labels = y[t][i][:, :-1], y[t][i][:, -1]
                boxes, labels = boxes.to(device), labels.to(device)
                box_dic = {'boxes': boxes, 'labels': labels}
                targets_.append(box_dic)

        features = {'level_'+str(i):item for i, item in enumerate(features)}
        image_size = (x.shape[-2], x.shape[-1])
        images = x.view(-1, *x.shape[2:])
        image_list = ImageList(images, [image_size]*len(images))

        losses = self._rpn.forward(image_list, features, targets_)[1]

        loss = sum(losses.values())
        return loss

