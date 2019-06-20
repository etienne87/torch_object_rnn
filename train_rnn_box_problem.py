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
from core.box import box_nms, box_iou, change_box_order
from core.ssd_loss import SSDLoss


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


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes, device=labels.device)  # [D,D]
    return y[labels]  # [N,D]


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
        self.default_boxes_xyxy =  change_box_order(self.default_boxes, 'xywh2xyxy')
        self.use_cuda = False
    
    def cuda(self):
          self.default_boxes = self.default_boxes.cuda()
          self.default_boxes_xyxy = self.default_boxes_xyxy.cuda()
          self.use_cuda = True    

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

        boxes = torch.cat(boxes, 0)

        return boxes, fm_sizes, fm_len

    def prepare_fmap_input(self, loc_targets, cls_targets, num_classes):
        r"""Use fmap sizes to encode
        loc_targets: TxN, D, 4
        cls_targets: TxN, D, 1
        """
        import pdb
        pdb.set_trace()
        n_anchors = len(self.aspect_ratios) * len(self.scale_ratios)
        cls_targets = one_hot_embedding(cls_targets, num_classes)
        loc = torch.cat((loc_targets, cls_targets), dim=2)
        out = torch.split(loc, self.fm_len, dim=1)
        res = []
        for fm_size, tensor in zip(self.fm_sizes, out):
            fm_h, fm_w = fm_size
            n, l, c = tensor.size()
            assert l == fm_h * fm_w
            ans = tensor.view(n, fm_h, fm_w, n_anchors * c)
            res.append(ans)
        return res

    def encode(self, boxes, labels):
        def argmax(x):
            v, i = x.max(0)
            j = v.max(0)[1].item() #was (0)[1][0] which causes Pytorch Warning for Scalars
            return (i[j], j)

        default_boxes = self.default_boxes_xyxy

        ious = box_iou(default_boxes, boxes)  # [#anchors, #obj]

        if self.use_cuda:
            index = torch.cuda.LongTensor(len(default_boxes)).fill_(-1)
        else:
            index = torch.LongTensor(len(default_boxes)).fill_(-1)

        masked_ious = ious.clone()
        while True:
            i, j = argmax(masked_ious)
            if masked_ious[i,j] < 1e-6:
                break
            index[i] = j
            masked_ious[i,:] = 0
            masked_ious[:,j] = 0

        mask = (index<0) & (ious.max(1)[0]>=0.5)
        if mask.any():
            index[mask] = ious[mask].max(1)[1]

        boxes = boxes[index.clamp(min=0)]  # negative index not supported
        boxes = change_box_order(boxes, 'xyxy2xywh')
        default_boxes = self.default_boxes

        loc_xy = (boxes[:,:2]-default_boxes[:,:2]) / default_boxes[:,2:] 
        loc_wh = torch.log(boxes[:,2:]/default_boxes[:,2:]) 
        loc_targets = torch.cat([loc_xy,loc_wh], 1)
        cls_targets = 1 + labels[index.clamp(min=0)]
        cls_targets[index<0] = 0
        return loc_targets, cls_targets

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
    def __init__(self, num_classes=2, num_fms=1):
        super(BoxTracker, self).__init__()
        nanchors = 9
        c = nanchors * (4 * nclasses)
        self.c = c
        self.cell = conv_lstm(c, 64)
        self.box = nn.Conv2d(64, c, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.rnn1(x)
        x = self.box(x)
        return x


def encode_boxes(targets, box_coder, cuda):
    loc_targets, cls_targets = [], []
    for t in range(len(targets)):
        for i in range(len(targets[t])):
            boxes, labels = targets[t][i][:, :-1], targets[t][i][:, -1]
            if cuda:
                boxes, labels = boxes.cuda(), labels.cuda()
            loc_t, cls_t = box_coder.encode(boxes, labels)
            loc_targets.append(loc_t.unsqueeze(0))
            cls_targets.append(cls_t.unsqueeze(0).long())
    loc_targets = torch.cat(loc_targets, dim=0)  # (N,#anchors,4)
    cls_targets = torch.cat(cls_targets, dim=0)  # (N,#anchors,C)
    if cuda:
        loc_targets = loc_targets.cuda()
        cls_targets = cls_targets.cuda()
        
    return loc_targets, cls_targets



if __name__ == '__main__':
    classes, cin, time, height, width = 2, 3, 10, 64, 64
    batchsize = 16
    epochs = 100
    cuda = False
    train_iter = 100
    nclasses = 2
    dataset = SquaresVideos(t=time, c=cin, h=height, w=width, batchsize=batchsize, normalize=False)
    dataset.num_frames = train_iter

    box_coder = BoxCoder(input_size=256, num_fms=1)
    seq = rnn.RNN(BoxTracker())

    if cuda:
        seq.cuda()
        box_coder.cuda()


    logdir = '/home/eperot/box_experiments/rnn_cv_combine_peephole/'
    writer = SummaryWriter(logdir)


    optimizer = optim.Adam(seq.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    criterion = SSDLoss(num_classes=2)

    proba = 1
    gt_every = 1
    alpha = 1

    for epoch in range(epochs):

        gt_every =    gt_every + epoch * 0.05
        proba *= 0.9
        alpha *= 0.9
        alpha = max(0.2, alpha)


        #train round
        print('TRAIN: PROBA RESET: ', proba)
        seq.reset()
        for batch_idx, data in enumerate(dataset):
            if np.random.rand() < proba:
                dataset.reset()
                seq.reset()

            _, targets = dataset.next()

            loc_targets, cls_targets = encode_boxes(targets, box_coder, cuda)


            fmaps = box_coder.prepare_fmap_input(loc_targets, cls_targets, nclasses)




            """
            batch = batch.cuda() if cuda else batch
            input, target = batch[:-1], batch[1:]
            optimizer.zero_grad()
            out = seq(input, alpha=alpha)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            if batch_idx%10 == 0:
                print('loss:', loss.item())
            writer.add_scalar('train_loss', loss.data.item(), batch_idx + epoch * len(dataset))
            """

        #val round: make video, initialize from real seq
        """
        print('DEMO:')
        with torch.no_grad():
            periods = 20
            nrows = 2
            ncols = batchsize / nrows
            seq.reset()
            grid = np.zeros((periods * time, nrows, ncols, dataset.height, dataset.width, 3), dtype=np.uint8)

            for period in range(periods):
                batch, _ = dataset.next()
                batch = batch.cuda() if cuda else batch
                out = seq(batch)
                images = out.cpu().data.numpy()
                for t in range(time):
                    for i in range(batchsize):
                        y = i / ncols
                        x = i % ncols
                        img = np.moveaxis(images[t, i], 0, 2)
                        img = (img-img.min())/(img.max()-img.min())
                        grid[t + period * time, y, x] = (img*255).astype(np.uint8)

            grid = grid.swapaxes(2, 3).reshape(periods * time, nrows * dataset.height, ncols * dataset.width, 3)
            utils.add_video(writer, 'test_with_xt', grid, global_step=epoch, fps=30)


            priods = 3
            grid = np.zeros((periods * time, nrows, ncols, dataset.height, dataset.width, 3), dtype=np.uint8)
            batch, _ = dataset.next()
            batch = batch.cuda() if cuda else batch
            out = seq(batch, future=periods*time)
            images = out.cpu().data.numpy()
            for t in range(periods*time):
                for i in range(batchsize):
                    y = i / ncols
                    x = i % ncols
                    img = np.moveaxis(images[t, i], 0, 2)
                    img = (img-img.min())/(img.max()-img.min())

                    grid[t, y, x] = (img*255).astype(np.uint8)
            grid = grid.swapaxes(2, 3).reshape(periods * time, nrows * dataset.height, ncols * dataset.width, 3)
            utils.add_video(writer, 'test_hallucinate', grid, global_step=epoch, fps=30)
        """

        scheduler.step()








