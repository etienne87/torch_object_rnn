from __future__ import print_function

import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import cv2

import core.recurrent as rnn
import core.utils as utils
from core.box_coder import SSDBoxCoder
from core.ssd_loss import SSDLoss


from tensorboardX import SummaryWriter
from functools import partial
from toy_pbm_detection import SquaresVideos


#Stupid check
"""
def stupid_check(dataset, box_coder, cuda, num_classes):
    _, targets = dataset.next()
    loc_targets, cls_targets = encode_boxes(targets, box_coder, cuda)
    fmaps = prepare_fmap_input(box_coder, loc_targets, cls_targets, nclasses, batchsize)
    loc_preds, cls_preds = [], []
    for fmap in fmaps:
        loc, cls = decode_boxes(fmap, num_classes, net.num_anchors)
        loc_preds.append(loc)
        cls_preds.append(cls)
    loc_preds = torch.cat(loc_preds, dim=1)
    cls_preds = torch.cat(cls_preds, dim=1)
    cls_preds = cls_preds.max(dim=-1)[1]
    test = loc_targets - loc_preds
    test2 = cls_targets - cls_preds
    assert test.sum().item() == 0
    assert test2.sum().item() == 0


def stupid_check_prepare_fmap_inputs(dataset, box_coder, cuda, num_classes):
    _, targets = dataset.next()
    loc_targets, cls_targets = encode_boxes(targets, box_coder, cuda)
    fmaps = prepare_fmap_input(box_coder, loc_targets, cls_targets, nclasses, batchsize)


def highlight_anchor_boxes_answering(cuda):
    size = 256
    dataset = SquaresVideos(t=5, c=3, h=size, w=size, batchsize=1, normalize=False)
    dataset.num_frames = 100

    net = ARSSD(2, size, size)
    box_coder = SSDBoxCoder(net)
    box_coder.iou_threshold = 0.5
    print(box_coder.print_info())
    if cuda:
        net.cuda()
        box_coder.cuda()

    while 1:
        _, targets = dataset.next()

        box_preds = []
        box_gt = []
        for t in range(len(targets[0])):
            boxes = targets[t][0][:, :-1]
            if cuda:
                boxes = boxes.cuda()
            anchors = box_coder.match(boxes)

            img = np.zeros((size,size,3), dtype=np.uint8)

            gt = boxes.cpu().numpy()
            x1, y1, x2, y2 = gt[0,0], gt[0,1], gt[0,2], gt[0,3]

            cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 2)

            if anchors is not None:
                anchors = anchors.cpu().numpy()
                for i in range(anchors.shape[0]):
                    x1, y1, x2, y2 = anchors[i,0], anchors[i,1], anchors[i,2], anchors[i,3]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

            cv2.imshow('img', img)
            cv2.waitKey(5)
"""

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
    oph = partial(nn.Conv2d, kernel_size=3, padding=1, stride=1, bias=False)
    #opx = partial(nn.Conv2d, kernel_size=3, padding=1, stride=1, bias=False)
    opx = partial(conv_bn, stride=stride)
    return rnn.LSTMCell(cin, cout, opx, oph, nonlinearity=torch.tanh)


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes, device=labels.device)  # [D,D]
    return y[labels]  # [N,D]


def prepare_fmap_input(box_coder, loc_targets, cls_targets, num_classes, batchsize):
    r"""Use fmap sizes to encode
    loc_targets: TxN, D, 4
    cls_targets: TxN, D, 1

    output packed T, N, (4+C), H, W to feed the network!
    """
    n_anchors = 2 * len(box_coder.aspect_ratios) + 2
    cls_targets = one_hot_embedding(cls_targets, num_classes)
    loc = torch.cat((loc_targets, cls_targets), dim=2)
    out = torch.split(loc, box_coder.fm_len, dim=1)
    res = []
    for fm_size, tensor in zip(box_coder.fm_sizes, out):
        fm_h, fm_w = fm_size
        n, l, c = tensor.size()
        assert l == fm_h * fm_w * n_anchors
        ans = tensor.view(n, fm_h, fm_w, n_anchors * c)
        ans = ans.permute([0, 3, 1, 2])
        ans = rnn.batch_to_time(ans, batchsize)
        res.append(ans)
    return res


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


def decode_boxes(box_map, num_classes, num_anchors):
    """
    box_map: T, N, C, H, W
    """
    fm_h, fm_w = box_map.shape[-2:]
    nboxes = fm_h * fm_w * num_anchors
    box_map = rnn.time_to_batch(box_map)
    #box_map = box_map.permute([0, 2, 3, 1]).contiguous().view(box_map.size(0), nboxes, 4+num_classes)

    #TN, C, H, W -> TN, Na, 4+C, H, W
    box_map = box_map.view(box_map.size(0), num_anchors, 4+num_classes, fm_h, fm_w)
    box_map = box_map.permute([0, 1, 3, 4, 2]).contiguous().view(box_map.size(0), nboxes, 4+num_classes)

    return box_map[..., :4], box_map[..., 4:]


def encode_boxes_fc(targets):
    x = []
    for t in range(len(targets)):
        boxes = []
        for i in range(len(targets[t])):
            box = targets[t][0][0,:-1] #1x4
            import pdb
            pdb.set_trace()
            boxes.append(box[None])
        batch = torch.cat(boxes, dim=1)
        x.append(batch[None])

    x = torch.cat(x, dim=0)


def remove_first(loc_targets, cls_targets, n):
    loc = rnn.batch_to_time(loc_targets, n)[1:]
    cls = rnn.batch_to_time(cls_targets, n)[1:]
    return rnn.time_to_batch(loc), rnn.time_to_batch(cls)


def remove_last(loc_targets, cls_targets, n):
    loc = rnn.batch_to_time(loc_targets, n)[:-1]
    cls = rnn.batch_to_time(cls_targets, n)[:-1]
    return rnn.time_to_batch(loc), rnn.time_to_batch(cls)


def get_box_params(sources, h, w):
    image_size = float(min(h, w))
    steps = []
    box_sizes = []
    fm_sizes = []
    s_min, s_max = 0.1, 0.9
    m = float(len(sources))
    for k, src in enumerate(sources):
        h2, w2 = src
        fm_sizes.append((h2, w2))
        step_y, step_x = math.floor(float(h) / h2), math.floor(float(w) / w2)
        steps.append((step_y, step_x))
        s_k = s_min + (s_max - s_min) * k / m
        box_sizes.append(math.floor(s_k * image_size))
    s_k = s_min + (s_max - s_min)
    box_sizes.append(s_k * image_size)
    return fm_sizes, steps, box_sizes



class BoxTracker(nn.Module):
    #this thing takes input image 
    #the RNN is fed with either CNN(image(t)) and a combination GT(t-1) and hidden(t-1)
    def __init__(self, nanchors, num_classes=2):
        super(BoxTracker, self).__init__()
        self.nanchors = nanchors
        self.num_classes = num_classes
        self.c = nanchors * (4 + num_classes)
        hidden = 64
        self.cell1 = conv_lstm(self.c, hidden)
        self.cell2 = conv_lstm(hidden, hidden)
        self.box = nn.Conv2d(hidden, self.c, 1, stride=1, padding=0, bias=False)

    def preprocess_output(self, ht, hard=False):
        #Chn: Chans/pixel = Nanchors x (4+C)
        #ht: (N, Chn, H, W)
        #ht is N, H, W, Nanchors x (4+C)
        n, chn, h, w = ht.size()
        ht = ht.detach()
        tmp = ht.view(n, self.nanchors, 4+self.num_classes, h, w)
        loc = tmp[:,:,:4,:,:]
        cls = F.softmax(tmp[:,:,4:,:,:], dim=2)

        if hard:
            amax = cls.max(dim=2)[1][:,:,None,:,:]
            o = torch.ones_like(amax).float()
            cls[...] = 0
            cls.scatter_(2, amax, o)

        tmp = torch.cat((loc, cls), dim=2).view(n, chn, h, w)
        return tmp

    def forward(self, x, alpha=1, future=0):
        self.detach_hidden()
        xseq = x.unbind(0)
        result = []
        ht = None
        xt = None

        # First treat sequence
        for t, xt in enumerate(xseq):

            v = random.uniform(0, 1)
            if ht is not None and v > alpha and self.time > 1:
                xt = self.preprocess_output(ht)

            #run network
            x = self.cell1(xt)
            x = self.cell2(x)
            ht = self.box(x)

            result.append(ht[None])
            self.time += 1

        # For auto-regressive use-cases
        if future:
            assert ht.shape[1] == xt.shape[1]
            for _ in range(future):
                xt = self.preprocess_output(ht)

                # run network
                x = self.cell1(xt)
                x = self.cell2(x)
                ht = self.box(x)

                result.append(ht[None])

        result = torch.cat(result, dim=0)
        return result

    def detach_hidden(self):
        self.cell1.detach_hidden()
        self.cell2.detach_hidden()

    def reset(self, mask=None):
        self.time = 0
        self.cell1.reset(mask)
        self.cell2.reset(mask)


class ARSSD(nn.Module):
    def __init__(self, num_classes, height, width):
        super(ARSSD, self).__init__()

        #here we simulate the image preliminary cnn
        sources = [(height/8, width/8), 
                    (height/16, width/16), 
                    (height/32, width/32)]

        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.fm_sizes, self.steps, self.box_sizes = get_box_params(sources, height, width)
        self.aspect_ratios = []
        self.num_anchors = 2 * len(self.aspect_ratios) + 2


        self.box_trackers = nn.ModuleList([
            BoxTracker(self.num_anchors, num_classes),
            BoxTracker(self.num_anchors, num_classes),
            BoxTracker(self.num_anchors, num_classes)
        ])

    def forward(self, xl, alpha=1.0, future=0):
        loc_preds, cls_preds = [], []
        for bt, x in zip(self.box_trackers, xl):
            y = bt(x, alpha, future) 
            loc, cls = decode_boxes(y, self.num_classes, self.num_anchors)
            loc_preds.append(loc)
            cls_preds.append(cls)

        loc_preds = torch.cat(loc_preds, dim=1)
        cls_preds = torch.cat(cls_preds, dim=1)
        return loc_preds, cls_preds

    def reset(self):
        for i in range(len(self.box_trackers)):
            self.box_trackers[i].reset()

if __name__ == '__main__':
    num_classes, cin, tbins, height, width = 2, 3, 10, 128, 128
    batchsize = 8
    epochs = 100
    cuda = 1
    train_iter = 300
    nclasses = 2
    dataset = SquaresVideos(t=tbins, c=cin, h=height, w=width,
                            batchsize=batchsize, max_classes=num_classes-1, render=False)
    dataset.num_frames = train_iter

    conv_encoding = True

    net = ARSSD(num_classes, height, width)
    box_coder = SSDBoxCoder(net)

    if cuda:
        net.cuda()
        box_coder.cuda()


    logdir = '/home/eperot/boxexp/tmp/'
    writer = SummaryWriter(logdir)


    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = SSDLoss(num_classes=2)
   
    proba = 1
    gt_every = 1
    alpha = 1

    for epoch in range(epochs):
        #train round
        print('TRAIN: PROBA RESET: ', proba, ' ALPHA: ', alpha)
        net.reset()
        for batch_idx, data in enumerate(dataset):
            if np.random.rand() < proba:
                dataset.reset()
                net.reset()
            
            _, targets = dataset.next()
            optimizer.zero_grad()


            loc_targets, cls_targets = encode_boxes(targets, box_coder, cuda)
            fmaps = prepare_fmap_input(box_coder, loc_targets, cls_targets, nclasses, batchsize)
            loc_targets, cls_targets = remove_first(loc_targets, cls_targets, batchsize)
            loc_preds, cls_preds = net(fmaps, alpha)
            loc_preds, cls_preds = remove_last(loc_preds, cls_preds, batchsize)
            loc_loss, cls_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss = loc_loss + cls_loss

            loss.backward()

            optimizer.step()
            if batch_idx%3 == 0:
                print('loss:', loss.item())
            writer.add_scalar('train_loss', loss.data.item(), batch_idx + epoch * len(dataset))
            

        #val round: make video, initialize from real seq
        print('DEMO:')

        with torch.no_grad():

            periods = 5
            nrows = 2
            ncols = batchsize / nrows
            net.reset()
            grid = np.zeros((periods * tbins, nrows, ncols, dataset.height, dataset.width, 3), dtype=np.uint8)

            for period in range(periods):
                _, targets = dataset.next()
                loc_targets, cls_targets = encode_boxes(targets, box_coder, cuda)
                fmaps = prepare_fmap_input(box_coder, loc_targets, cls_targets, nclasses, batchsize)
                loc_preds, cls_preds = net(fmaps)

                loc_preds = rnn.batch_to_time(loc_preds, batchsize)
                cls_preds = rnn.batch_to_time(cls_preds, batchsize)
                for t in range(tbins):
                    for i in range(batchsize):
                        y = i / ncols
                        x = i % ncols
                        img = grid[t + period * tbins, y, x]

                        boxes, labels, scores = box_coder.decode(loc_preds[t, i].data,
                                                                 F.softmax(cls_preds[t, i].data, dim=-1),
                                                                  nms_thresh=0.6,
                                                                  score_thresh=0.3)
                        if boxes is not None:
                            bboxes = utils.boxarray_to_boxes(boxes, labels, dataset.labelmap)
                            img = utils.draw_bboxes(img, bboxes)

                        grid[t + period * tbins, y, x] = img


            grid = grid.swapaxes(2, 3).reshape(periods * tbins, nrows * dataset.height, ncols * dataset.width, 3)
            utils.add_video(writer, 'test_with_xt', grid, global_step=epoch, fps=30)

            periods = 5
            grid = np.zeros((periods * tbins, nrows, ncols, dataset.height, dataset.width, 3), dtype=np.uint8)
            batch, _ = dataset.next()
            _, targets = dataset.next()
            loc_targets, cls_targets = encode_boxes(targets, box_coder, cuda)
            fmaps = prepare_fmap_input(box_coder, loc_targets, cls_targets, nclasses, batchsize)
            loc_preds, cls_preds = net(fmaps, future=int(periods*tbins))
            loc_preds = rnn.batch_to_time(loc_preds, batchsize)
            cls_preds = rnn.batch_to_time(cls_preds, batchsize)
            for t in range(periods*tbins):
                for i in range(batchsize):
                    y = i / ncols
                    x = i % ncols
                    img = grid[t, y, x]

                    boxes, labels, scores = box_coder.decode(loc_preds[t, i].data,
                                                             F.softmax(cls_preds[t, i].data, dim=-1),
                                                              nms_thresh=0.6,
                                                              score_thresh=0.3)
                    if boxes is not None:
                        bboxes = utils.boxarray_to_boxes(boxes, labels, dataset.labelmap)
                        img = utils.draw_bboxes(img, bboxes)

                    grid[t, y, x] = img

            grid = grid.swapaxes(2, 3).reshape(periods * tbins, nrows * dataset.height, ncols * dataset.width, 3)
            utils.add_video(writer, 'test_hallucinate', grid, global_step=epoch, fps=30)

        scheduler.step()
        proba *= 0.9
        alpha *= 0.9
        alpha = max(0.7, alpha)








