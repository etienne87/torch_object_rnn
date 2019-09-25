from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.ssd.box_coder import SSDBoxCoder
from core.ssd.loss import SSDLoss
from core.modules import FPN, Trident
import math




def get_box_params(sources, h, w):
    image_size = float(min(h, w))
    steps = []
    box_sizes = []
    fm_sizes = []
    s_min, s_max = 0.1, 0.9
    m = float(len(sources))
    for k, src in enumerate(sources):
        # featuremap size
        fm_sizes.append((src.size(2), src.size(3)))

        # step is ratio image_size / featuremap_size
        step_y, step_x = math.floor(float(h) / src.size(2)), math.floor(float(w) / src.size(3))
        steps.append((step_y, step_x))

        # compute scale
        s_k = s_min + (s_max - s_min) * k / m

        # box_size is scale * image_size
        box_sizes.append(math.floor(s_k * image_size))

        print("box size: ", box_sizes[-1])

    s_k = s_min + (s_max - s_min)
    box_sizes.append(s_k * image_size)

    return fm_sizes, steps, box_sizes



class SSD(nn.Module):
    def __init__(self, feature_extractor=Trident,
                 num_classes=2, cin=2, height=300, width=300, act='softmax', shared=False):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.height, self.width = height, width
        self.cin = cin

        self.extractor = feature_extractor(cin)

        x = torch.randn(1, 1, self.cin, self.height, self.width)
        sources = self.extractor(x)

        self.fm_sizes, self.steps, self.box_sizes = get_box_params(sources, self.height, self.width)
        self.ary = float(width) / height

        self.aspect_ratios = []
        self.in_channels = [item.size(1) for item in sources]
        self.num_anchors = [2 * len(self.aspect_ratios) + 2 for i in range(len(self.in_channels))]

        self.shared = shared
        self.cls_layers = nn.ModuleList()
        self.reg_layers = nn.ModuleList()
        if self.shared:
            self.reg_layers += [nn.Conv2d(self.in_channels[0], self.num_anchors[0] * 4,
                                          kernel_size=3, padding=1, stride=1)]
            self.cls_layers += [nn.Conv2d(self.in_channels[0], self.num_anchors[0] * self.num_classes,
                                          kernel_size=3, padding=1, stride=1)]
        else:
            for i in range(len(self.in_channels)):
                self.reg_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i]*4,
                                                   kernel_size=3, padding=1, stride=1)]
                self.cls_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i]*self.num_classes,
                                                   kernel_size=3, padding=1, stride=1)]


        self.act = act
        self.box_coder = SSDBoxCoder(self, 0.4, 0.7)
        self.criterion = SSDLoss(num_classes=num_classes,
                                 mode='ohem',
                                 use_sigmoid=self.act=='sigmoid',
                                 use_iou=False)

        for l in self.reg_layers:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

        # Init for strong bias toward bg class for focal loss
        if self.act == 'softmax':
            self.softmax_init()
        else:
            self.sigmoid_init()

    def resize(self, height, width):
        self.height, self.width = height, width
        self.extractor.reset()
        x = torch.randn(1, 1, self.cin, self.height, self.width).to(self.box_coder.default_boxes.device)
        sources = self.extractor(x)
        self.fm_sizes, self.steps, self.box_sizes = get_box_params(sources, self.height, self.width)
        self.ary = float(width) / height
        self.box_coder.reset(self)
        self.extractor.reset()

    def sigmoid_init(self):
        px = 0.99
        bias_bg = math.log(px / (1 - px))
        for i, l in enumerate(self.cls_layers):
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)
            l.bias.data = l.bias.data.reshape(self.num_anchors[i], self.num_classes)
            l.bias.data[:, 0:] -= bias_bg
            l.bias.data = l.bias.data.reshape(-1)

    def softmax_init(self):
        px = 0.99
        K = self.num_classes - 1
        bias_bg = math.log(K * px / (1 - px))
        for i, l in enumerate(self.cls_layers):
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)
            l.bias.data = l.bias.data.reshape(self.num_anchors[i], self.num_classes)
            l.bias.data[:, 0] += bias_bg
            l.bias.data = l.bias.data.reshape(-1)

    def reset(self):
        self.extractor.reset()

    def _print_average_scores_fg_bg(self, cls_preds):
        # Verbose check of Average Probability
        scores = F.softmax(cls_preds.data, dim=-1)
        bg_score = scores[:, :, 0].mean().item()
        fg_score = scores[:, :, 1:].sum(dim=-1).mean().item()
        print('Average BG_Score: ', bg_score, 'Averag FG_Score: ', fg_score)

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        xs = self.extractor(x)
        for i, x in enumerate(xs):
            if self.shared:
                loc_pred = self.reg_layers[0](x)
                cls_pred = self.cls_layers[0](x)
            else:
                loc_pred = self.reg_layers[i](x)
                cls_pred = self.cls_layers[i](x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()

            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))
            cls_preds.append(cls_pred.view(cls_pred.size(0), -1, self.num_classes))

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)

        # self._print_average_scores_fg_bg(cls_preds)
        if not self.training:
            if self.act == 'softmax':
                cls_preds = F.softmax(cls_preds, dim=2)
            else:
                cls_preds = torch.sigmoid(cls_preds)

        return loc_preds, cls_preds

    def compute_loss(self, x, targets):
        loc_preds, cls_preds = self(x)
        loc_targets, cls_targets = self.box_coder.encode_txn_boxes(targets)

        if self.criterion.use_iou:
            loc_preds = self.box_coder.decode_loc(loc_preds)
            loc_targets = self.box_coder.decode_loc(loc_targets)

        loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)

        loss = loc_loss + cls_loss
        return {'total':loss, 'loc': loc_loss, 'cls_loss': cls_loss}

    def get_boxes(self, x, score_thresh=0.4):
        loc_preds, cls_preds = self(x)
        targets = self.box_coder.decode_txn_boxes(loc_preds, cls_preds, x.size(1), score_thresh=score_thresh)
        return targets


if __name__ == '__main__':
    net = FPN(1)
    t, n, c, h, w = 10, 1, 1, 128, 128
    x = torch.rand(t, n, c, h, w)
    y = net(x)
