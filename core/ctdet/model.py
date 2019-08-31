'''
https://arxiv.org/pdf/1904.07850.pdf: CenterNet
https://arxiv.org/pdf/1808.01244.pdf: CornerNet
'''
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


def decode_boxes(box_map, num_classes):
    box_map = box_map.permute([0, 2, 3, 1]).contiguous().view(box_map.size(0), -1, 4 + num_classes)
    return box_map[..., :4], box_map[..., 4:]


class CenterNet(nn.Module):
    def __init__(self, feature_extractor, num_classes=2, cin=2, height=300, width=300, act='softmax'):
        super(CenterNet, self).__init__()
        self.num_classes = num_classes
        self.height, self.width = height, width
        self.cin = cin
        self.extractor = feature_extractor(cin)
        self.act = (F.softmax if act == "softmax" else torch.sigmoid) if not self.training else lambda x:x

        x = torch.randn(1, 1, self.cin, self.height, self.width)
        y = self.extractor(x)
        c, h, w = y.shape[-3:]
        self.stride = max(self.height / h, self.width / w)
        self.pred = nn.Conv2d(c, (4+self.num_classes), kernel_size=3, padding=1, stride=1)

    def reset(self):
        self.extractor.reset()

    def forward(self, x):
        y = self.extractor(x)
        out = self.pred(x)
        out = out.permute([0, 2, 3, 1]).contiguous().view(out.size(0), -1, 4 + self.num_classes)
        loc, cls = out[..., :4], out[..., 4:]
        cls = self.act(cls)
        loc = loc.view(loc.size(0), -1, 4)
        cls = self.act(cls.view(cls.size(0), -1, self.num_classes))
        return loc, cls

    def fit(self, x, boxes):
        return Exception("Not Implemented yet")
        loc, cls = self.forward(x)

        n, t = x.shape[:2]
        h, w = self.height/self.stride, self.width/self.stride
        loc_t = torch.zeros((n, t, 4, h, w), dtype=torch.float32, device=x.device)
        cls_t = torch.zeros((n, t, h, w), dtype=torch.float32, device=x.device)

        #splat onto cls_t gaussians of sigmoid min(w/2, h/2)/3
        for t in range(len(boxes)):
            for i in range(len(boxes[t])):
                box, labels = boxes[t][i][:, :-1], boxes[t][i][:, -1]

                xc, yc = (box[:, 0] + box[:, 2])/2, (box[:, 1] + box[:, 3])/2
                xc_prime, yc_prime = xc/self.stride, yc/self.stride
                #TODO:
                # splat gaussian

                pass

        loc_t = loc_t.view(loc_t.size(0), -1, 4)
        cls_t = cls_t.view(cls_t.size(0), -1)

        return loc, cls, loc_t, cls_t
