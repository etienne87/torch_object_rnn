from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn

import numpy as np
import math

from core.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from core.utils.image import draw_dense_reg

from core.modules import UNet





class Options:
    def __init__(self, max_objs=5, pad=0, reg_offset=True,
                       dense_wh=True, cat_spec_wh=False, cat_spec_mask=False):
        self.max_objs = max_objs
        self.pad = pad
        self.dense_wh = dense_wh
        self.cat_spec_wh = cat_spec_wh
        self.cat_spec_mask = cat_spec_mask
        self.reg_offset = reg_offset


class CenterNet(nn.Module):
    def __init__(self, backbone=UNet,
                        num_classes=2, cin=2, height=300, width=300):
        super(CenterNet, self).__init__()
        self.num_classes = num_classes
        self.height, self.width = height, width
        self.cin = cin
        self.backbone = backbone(cin)
        x = torch.randn(1, 1, self.cin, self.height, self.width)
        y = self.backbone(x)
        c, h, w = y.shape[-3:]
        self.stride = max(self.height / h, self.width / w)
        self.pred = nn.Conv2d(c, (4+self.num_classes), kernel_size=3, padding=1, stride=1)

        self.max_objs = 5
        self.pad = 1
        self.dense_wh = True
        self.cat_spec_wh = True
        self.cat_spec_mask = False
        self.reg_offset = True


    def reset(self):
        self.extractor.reset()

    def forward(self, x):
        y = self.backbone(x)
        out = self.pred(x)
        out = out.permute([0, 2, 3, 1]).contiguous().view(out.size(0), -1, 4 + self.num_classes)
        loc, cls = out[..., :4], out[..., 4:]
        loc = loc.view(loc.size(0), -1, 4)
        cls = self.act(cls.view(cls.size(0), -1, self.num_classes))
        return loc, cls

    def compute_loss(self, x, targets):
        loc_preds, cls_preds = self(x)
        loc_targets, cls_targets = self.encode_txn_boxes(targets)
        loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss = loc_loss + cls_loss
        return loss

    def encode_txn_boxes(self, x, targets):
        t, n = x.shape[:2]
        h, w = self.height/self.stride, self.width/self.stride
        cls = torch.zeros((n, t, 4, h, w), dtype=np.float32)
        reg = torch.zeros((n, t, h, w), dtype=torch.float32, device=x.device)

        #splat onto cls_t gaussians of sigmoid min(w/2, h/2)/3
        for t in range(len(targets)):
            for n in range(len(targets[t])):
                boxes, labels = targets[t][n][:, :-1], targets[t][n][:, -1]
                ret = CenterNet.make_gt_maps(boxes, self.num_classes, self.height, self.width, self.opt)
                #fill cls & reg

        return cls, reg

    def make_gt_maps(self, boxes, num_classes, height, width):
        import pdb
        pdb.set_trace()
        c = np.array([width / 2., height / 2.], dtype=np.float32)

        input_h = (height | self.pad) + 1
        input_w = (width | self.pad) + 1
        s = np.array([input_w, input_h], dtype=np.float32)

        output_h = input_h // self.down_ratio
        output_w = input_w // self.down_ratio
        num_classes = num_classes

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((opt.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((opt.max_objs, 2), dtype=np.float32)
        ind = np.zeros((opt.max_objs), dtype=np.int64)
        reg_mask = np.zeros((opt.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((opt.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((opt.max_objs, num_classes * 2), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if opt.mse_loss else \
            draw_umich_gaussian

        num_objs = boxes.size(0)

        gt_det = []
        for k in range(num_objs):
            bbox, cls_id = boxes[k, :-1], boxes[k, -1]
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = opt.hm_gauss if opt.mse_loss else radius
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

        ret = {'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        if opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if opt.reg_offset:
            ret.update({'reg': reg})

        return ret
