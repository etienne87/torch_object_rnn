from __future__ import print_function

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math


import core.recurrent as rnn
import core.utils as utils
from core.box import box_nms, box_iou



from tensorboardX import SummaryWriter
from functools import partial
from toy_pbm_detection import SquaresVideos


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

def conv_lstm(cin, cout, stride=1):
    oph = partial(nn.Conv2d, kernel_size=3, padding=1, stride=1, bias=True)
    opx = partial(conv_bn, stride=stride)
    return rnn.LSTMCell(cin, cout, opx, oph, nonlinearity=torch.tanh)


def meshgrid(x, y, row_major=True):
    a = torch.arange(0, x).float()
    b = torch.arange(0, y).float()
    xx = a.repeat(y).view(-1,1)
    yy = b.view(-1,1).repeat(1,x).view(-1,1)
    return torch.cat([xx,yy],1) if row_major else torch.cat([yy,xx],1)


class BoxCoder:
    """Encode a list of feature map boxes
    """
    def __init__(self, input_size=128, num_fms=4):
        self.anchor_sizes = 2**np.linspace(5, np.log(input_size)/np.log(2), num_fms)
        self.anchor_areas = [item*item for item in self.anchor_sizes.tolist()]
        self.aspect_ratios = (1/2., 1/1., 2/1.)
        self.scale_ratios = (1., pow(2,1/3.), pow(2,2/3.))

        input_size = torch.tensor([float(input_size), float(input_size)])
        self.default_boxes, self.fm_sizes, self.fm_len = self._get_default_boxes(input_size=input_size)

    def _get_anchor_wh(self):
        '''Compute anchor width and height for each feature map.
        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar
                h = math.sqrt(s/ar)
                w = ar * h
                for sr in self.scale_ratios:  # scale
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.tensor(anchor_wh).view(num_fms,-1, 2)

    def _get_default_boxes(self, input_size):
        num_fms = len(self.anchor_areas)
        anchor_wh = self._get_anchor_wh()
        fm_sizes = [(input_size/pow(2.,i+3)).ceil() for i in range(num_fms)]

        print('fm sizes: ', fm_sizes)
        boxes = []
        fm_len = []
        size = 0
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w,fm_h) + 0.5  # [fm_h*fm_w, 2]
            xy = (xy * grid_size).view(fm_h,fm_w,1,2).expand(fm_h,fm_w,9,2)
            wh = anchor_wh[i].view(1,1,9,2).expand(fm_h,fm_w,9,2)
            box = torch.cat([xy-wh/2.,xy+wh/2.], 3)  # [x,y,x,y]
            box = box.view(-1,4)
            boxes.append(box)
            fm_len.append(len(box))

        return boxes, fm_sizes, fm_len

    def prepare_fmap_input(self, loc_targets, loc_cls):
        r"""Use fmap sizes to encode
        """
        loc = torch.cat((loc_targets, loc_cls), dim=2)
        out = torch.split(loc, self.fm_len)
        res = []
        for fm_size, tensor in zip(self.fm_sizes, out):
            n, l, c = tensor.size()
            assert l == fm_h * fm_w
            ans = tensor.view(n, fm_h, fm_w, c)
            res.append(ans)
        return res

    def encode(self, boxes):
        def argmax(x):
            v, i = x.max(0)
            j = v.max(0)[1].item() #was (0)[1][0] which causes Pytorch Warning for Scalars
            return (i[j], j)

        default_boxes = self.default_boxes_xyxy

        ious = box_iou(default_boxes, boxes)  # [#anchors, #obj]

        if self.use_cuda:
            index = torch.cuda.LongTensor(len(default_boxes)).fill_(-1)
            weights = torch.cuda.FloatTensor(len(default_boxes)).fill_(1.0)
        else:
            index = torch.LongTensor(len(default_boxes)).fill_(-1)
            weights = torch.FloatTensor(len(default_boxes)).fill_(1.0)

        masked_ious = ious.clone()
        while True:
            i, j = argmax(masked_ious)
            if masked_ious[i,j] < 1e-6:
                break
            index[i] = j
            masked_ious[i,:] = 0
            masked_ious[:,j] = 0

        mask = (index<0) & (ious.max(1)[0]>=self.iou_threshold)
        if mask.any():
            weights[mask], index[mask] = ious[mask].max(1)

        boxes = boxes[index.clamp(min=0)]  # negative index not supported
        boxes = change_box_order(boxes, 'xyxy2xywh')
        default_boxes = self.default_boxes # change_box_order(default_boxes, 'xyxy2xywh')

        loc_xy = (boxes[:,:2]-default_boxes[:,:2]) / default_boxes[:,2:] / self.variances[0]
        loc_wh = torch.log(boxes[:,2:]/default_boxes[:,2:]) / self.variances[1]
        loc_targets = torch.cat([loc_xy,loc_wh], 1)
        cls_targets = 1 + labels[index.clamp(min=0)]
        cls_targets[index<0] = 0
        return loc_targets, cls_targets, weights

    def decode(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        xy = loc_preds[:,:2] * self.variances[0] * self.default_boxes[:,2:] + self.default_boxes[:,:2]
        wh = torch.exp(loc_preds[:,2:] * self.variances[1]) * self.default_boxes[:,2:]
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

            if nms_thresh == 1.0:
                boxes.append(box)
                labels.append(torch.LongTensor(len(box)).fill_(i))
                scores.append(score)
            else:
                keep = box_nms(box, score, nms_thresh)
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



class BoxTracker(nn.Module):
    #this thing takes input image 
    #the RNN is fed with either CNN(image(t)) and a combination GT(t-1) and hidden(t-1)
    def __init__(self, input_size, num_fms=4):
        super(BoxTracker, self).__init__()
        self.cnn = rnn.SequenceWise(
                    nn.Sequential(
                    conv_bn(3, 8, 2),
                    conv_bn(8, 16, 2),
                    conv_bn(16, 32, 2)
                    )
            )

        self.rnn1 = 



if __name__ == '__main__':
    """
    classes, cin, time, height, width = 2, 3, 10, 64, 64
    batchsize = 16
    epochs = 100
    cuda = True
    train_iter = 100
    dataset = SquaresVideos(t=time, c=cin, h=height, w=width, batchsize=batchsize, normalize=False)
    dataset.num_frames = train_iter
    """

    boxcoder = BoxCoder(input_size=256, num_fms=1)