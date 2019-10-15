from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.ssd.box_coder import SSDBoxCoder, get_box_params_fixed_size
from core.ssd.loss import SSDLoss
from core.backbones import FPN
from core.anchors import Anchors
import math





USE_ANCHOR_MODULE = False

class SSD(nn.Module):
    def __init__(self, feature_extractor=FPN,
                 num_classes=2, cin=2, height=300, width=300, act='sigmoid', shared=True):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.height, self.width = height, width
        self.cin = cin

        self.extractor = feature_extractor(cin)

        x = torch.randn(1, 1, self.cin, self.height, self.width)
        sources = self.extractor(x)

        if USE_ANCHOR_MODULE:
            self.box_coder = Anchors(pyramid_levels=[i for i in range(3,3+len(sources))],
                                     scales=[1.0, 1.5],
                                     ratios=[1],
                                     label_offset=1,
                                     fg_iou_threshold=0.7, bg_iou_threshold=0.4)

            self.num_anchors = self.box_coder.num_anchors
        else:
            self.fm_sizes, self.steps, self.box_sizes = get_box_params_fixed_size(sources, height, width)
            self.ary = float(width) / height
            self.aspect_ratios = [1]
            self.scales = [1, 1.5]
            self.num_anchors = len(self.aspect_ratios) * len(self.scales) # self.num_anchors = 2 * len(self.aspect_ratios) + 2
            self.box_coder = SSDBoxCoder(self, 0.4, 0.7)

        self.aspect_ratios = []
        self.in_channels = [item.size(1) for item in sources]

        self.shared = shared
        self.act = act

        self.use_embedding_loss = False

        if self.shared:
            self.embedding_dims = 32

            self.loc_head = self._make_head(self.in_channels[0], self.num_anchors * 4)
            self.cls_head = self._make_head(self.in_channels[0], self.num_anchors * self.num_classes)
            if self.use_embedding_loss:
                self.emb_head = self._make_head(self.in_channels[0], self.num_anchors * self.embedding_dims)

            torch.nn.init.normal_(self.loc_head[-1].weight, std=0.01)
            torch.nn.init.constant_(self.loc_head[-1].bias, 0)

            if self.act == 'softmax':
                self.softmax_init(self.cls_head[-1])
            else:
                self.sigmoid_init(self.cls_head[-1])

        else:
            self.cls_layers = nn.ModuleList()
            self.reg_layers = nn.ModuleList()

            for i in range(len(self.in_channels)):
                self.reg_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors*4,
                                                   kernel_size=3, padding=1, stride=1)]
                self.cls_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors*self.num_classes,
                                                   kernel_size=3, padding=1, stride=1)]

                for l in self.reg_layers:
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

                # Init for strong bias toward bg class for focal loss
                if self.act == 'softmax':
                    self.softmax_init(self.cls_layers[-1])
                else:
                    self.sigmoid_init(self.cls_layers[-1])


        self.criterion = SSDLoss(num_classes=num_classes,
                                 mode='focal',
                                 use_sigmoid=self.act=='sigmoid',
                                 use_iou=False)

        self._forward = [self._forward_unshared, self._forward_shared][shared]


    # def resize(self, height, width):
    #     self.height, self.width = height, width
    #     self.extractor.reset()
    #     x = torch.randn(1, 1, self.cin, self.height, self.width).to(self.box_coder.default_boxes.device)
    #     sources = self.extractor(x)
    #     self.fm_sizes, self.steps, self.box_sizes = get_box_params_fixed_size(sources, self.height, self.width)
    #     self.ary = float(width) / height
    #     self.box_coder.reset(self)
    #     self.extractor.reset()

    def sigmoid_init(self, l):
        px = 0.99
        bias_bg = math.log(px / (1 - px))
        torch.nn.init.normal_(l.weight, std=0.01)
        torch.nn.init.constant_(l.bias, 0)
        l.bias.data = l.bias.data.reshape(self.num_anchors, self.num_classes)
        l.bias.data[:, 0:] -= bias_bg
        l.bias.data = l.bias.data.reshape(-1)

    def softmax_init(self, l):
        px = 0.99
        K = self.num_classes - 1
        bias_bg = math.log(K * px / (1 - px))
        torch.nn.init.normal_(l.weight, std=0.01)
        torch.nn.init.constant_(l.bias, 0)
        l.bias.data = l.bias.data.reshape(self.num_anchors, self.num_classes)
        l.bias.data[:, 0] += bias_bg
        l.bias.data = l.bias.data.reshape(-1)

    def _make_head(self, in_planes, out_planes):
        layers = []
        layers.append(nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(True))

        for _ in range(0):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))

        return nn.Sequential(*layers)


    def reset(self):
        self.extractor.reset()

    def _print_average_scores_fg_bg(self, cls_preds):
        # Verbose check of Average Probability
        scores = F.softmax(cls_preds.data, dim=-1)
        bg_score = scores[:, :, 0].mean().item()
        fg_score = scores[:, :, 1:].sum(dim=-1).mean().item()
        print('Average BG_Score: ', bg_score, 'Averag FG_Score: ', fg_score)

    def _apply_head(self, layer, xs, ndims):
        out = []
        for x in xs:
            y = layer(x).permute(0, 2, 3, 1).contiguous()
            y = y.view(y.size(0), -1, ndims)
            out.append(y)
        out = torch.cat(out, 1)
        return out

    def _forward_shared(self, xs):
        loc_preds = self._apply_head(self.loc_head, xs, 4)
        cls_preds = self._apply_head(self.cls_head, xs, self.num_classes)

        if not self.training:
            if self.act == 'softmax':
                cls_preds = F.softmax(cls_preds, dim=2)
            else:
                cls_preds = torch.sigmoid(cls_preds)

        out_dic = {'loc':loc_preds, 'cls': cls_preds}

        if self.use_embedding_loss:
            out_dic['emb'] = self._apply_head(self.emb_head, xs, self.embedding_dims)

        return out_dic

    def _forward_unshared(self, xs):
        loc_preds = []
        cls_preds = []
        for i, x in enumerate(xs):
            loc_pred = self.reg_layers[i](x)
            cls_pred = self.cls_layers[i](x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()

            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))
            cls_preds.append(cls_pred.view(cls_pred.size(0), -1, self.num_classes))

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)

        if not self.training:
            if self.act == 'softmax':
                cls_preds = F.softmax(cls_preds, dim=2)
            else:
                cls_preds = torch.sigmoid(cls_preds)

        return {'loc':loc_preds, 'cls': cls_preds}

    def forward(self, x):
        xs = self.extractor(x)
        return self._forward(xs)

    def compute_loss(self, x, targets):
        xs = self.extractor(x)
        out_dic = self._forward(xs)

        if USE_ANCHOR_MODULE:
            loc_targets, cls_targets = self.box_coder.encode_txn_boxes(xs, targets)
        else:
            loc_targets, cls_targets = self.box_coder.encode_txn_boxes(targets)

        loc_preds, cls_preds = out_dic['loc'], out_dic['cls']

        loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)

        loss_dict = {'loc': loc_loss, 'cls_loss': cls_loss}


        return loss_dict

    def get_boxes(self, x, score_thresh=0.4):
        xs = self.extractor(x)
        out_dic = self._forward(xs)
        loc_preds, cls_preds = out_dic['loc'], out_dic['cls']
        if USE_ANCHOR_MODULE:
            targets = self.box_coder.decode_txn_boxes(xs, loc_preds, cls_preds, x.size(1), score_thresh=score_thresh)
        else:
            targets = self.box_coder.decode_txn_boxes(loc_preds, cls_preds, x.size(1), score_thresh=score_thresh)
        return targets


if __name__ == '__main__':
    net = FPN(1)
    t, n, c, h, w = 10, 1, 1, 128, 128
    x = torch.rand(t, n, c, h, w)
    y = net(x)
