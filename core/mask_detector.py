from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.losses import reduce
from core.backbones import FPN, MobileNetFPN, ResNet50FPN, ResNet50SSD
from datasets.multimodal import multimodal_annotate

def sigmoid_focal_loss(x, y, reduction='none'):
    ''' sigmoid focal loss

    :param x: [N, C, H, W]
    :param y: [N, H, W] (0: background, [1,C]: classes)
    :param reduction:
    :return:
    '''
    alpha = 0.25
    gamma = 2.0
    s = x.sigmoid()
    batchsize, num_classes, height, width = x.shape

    y2 = y.unsqueeze(2)
    fg = (y2>0).to(x)
    y_index = (y2 - 1).clamp_(0)
    t = torch.zeros((batchsize, num_anchors, height, width), dtype=x.dtype, device=x.device)
    t.scatter_(2, y_index, fg)

    pt = (1 - s) * t + s * (1 - t)
    focal_weight = (alpha * t + (1 - alpha) *
                    (1 - t)) * pt.pow(gamma)

    # focal_weight = pt.pow(gamma)

    loss = F.binary_cross_entropy_with_logits(
        x, t, reduction='none') * focal_weight
    loss = loss.sum(dim=-1)
    loss[y < 0] = 0

    loss = reduce(loss, reduction)
    return loss

def add_mask(videos, boxes, shift=0):
    time, batchsize, _, height, width = videos.shape[-2:]
    height2, width2 = height>>shift, width>>shift
    masks = {}
    masks['i'] = torch.zeros((time, batchsize, height2, width2), dtype=torch.int32)
    masks['b'] = torch.zeros((time, batchsize, height2, width2), dtype=torch.int32)
    masks['o'] = torch.zeros((time, batchsize, height2, width2), dtype=torch.int32)
    masks['s'] = torch.zeros((time, batchsize, height2, width2), dtype=torch.int32)
    for n in range(batchsize):
        for t in range(time):
            i,b,o,s = multimodal_annotate(boxes[t][n].numpy(), height, width, shift=shift)
            masks['i'][t, n] = torch.from_numpy(i)
            masks['b'][t, n] = torch.from_numpy(b)
            masks['o'][t, n] = torch.from_numpy(o)
            masks['s'][t, n] = torch.from_numpy(s)
    return masks


class MaskDetector(nn.Module):
    def __init__(self, feature_extractor=FPN, 
                 in_channels=3, 
                 num_classes=2,
                 act='sigmoid'):
        super(MaskDetector, self).__init__()
        self.label_offset = 1 * (act=='softmax')
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.act = act
        self.feature_extractor = feature_extractor(in_channels)
        self.criterion = sigmoid_focal_loss
        self.mask = nn.Sequential(
            nn.Conv2d(self.feature_extractor.cout, self.num_classes + self.label_offset, 1, 1, 0),
            nn.UpsamplingBilinear2d(scale_factor=8)
        )

    def reset(self):
        self.feature_extractor.reset()

    def forward(self, x):
        src = self.feature_extractor(x)[-1]
        return self.mask(src)

    def compute_loss(self, x, targets):
        masmasksk = self(x)
        semantics = self.criterion(masks, target_mask)
        loss_dict = {'semantics': semantics}
        return loss_dict

    def get_boxes(self, x, score_thresh=0.4):
        masks = self(x).sigmoid()
        targets = self.find_boxes(masks)
        return targets

    @classmethod
    def tiny_rnn_fpn(cls, in_channels, num_classes, act='sigmoid', loss='_focal_loss'):
        return cls(FPN, in_channels, num_classes, act, ratios=[1.0], scales=[1.0, 1.5], loss=loss)
