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
from functools import partial

import core.recurrent as rnn
import core.utils as utils
from tensorboardX import SummaryWriter
from toy_pbm_detection import SquaresVideos


def encode_boxes_fc(targets):
    x = []
    for t in range(len(targets)):
        boxes = []
        for i in range(len(targets[t])):
            box = targets[t][i][0,:-1] #1x4
            boxes.append(box[None])
        batch = torch.cat(boxes, dim=0)
        x.append(batch[None])
    x = torch.cat(x, dim=0)
    return x


class Sequence(nn.Module):
    def __init__(self, batchsize):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(4, 128)
        self.lstm2 = nn.LSTMCell(128, 128)
        self.linear = nn.Linear(128, 4)
        self.batchsize = batchsize

        self.register_buffer('h_t', torch.zeros(self.batchsize, 128, dtype=torch.float))
        self.register_buffer('c_t', torch.zeros(self.batchsize, 128, dtype=torch.float))
        self.register_buffer('h_t2', torch.zeros(self.batchsize, 128, dtype=torch.float))
        self.register_buffer('c_t2', torch.zeros(self.batchsize, 128, dtype=torch.float))

    def forward(self, input, future = 0, alpha=1):
        outputs = []
        output = None
        h_t, c_t = self.h_t.detach(), self.c_t.detach()
        h_t2, c_t2 = self.h_t2.detach(), self.c_t2.detach()

        for i, input_t in enumerate(input.unbind(0)):

            if output is not None:
                if np.random.rand() > max(0.1, alpha):
                    input_t = output.detach()

            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output[None]]

        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output[None]]

        self.h_t, self.c_t = h_t, c_t
        self.h_t2, self.c_t2 = h_t2, c_t2

        outputs = torch.cat(outputs, 0)
        return outputs

    def reset(self):
        self.h_t[...] = 0
        self.c_t[...] = 0
        self.h_t2[...] = 0
        self.c_t2[...] = 0

def ln_lin(cin, cout):
    return nn.Sequential(nn.LayerNorm(cin), nn.Linear(cin, cout, bias=True))

def bn_lin(cin, cout):
    return nn.Sequential(nn.BatchNorm1d(cin), nn.Linear(cin, cout, bias=True))

def fc_lstm(cin, cout):
    oph = partial(nn.Linear, bias=True)
    opx = partial(bn_lin)
    return rnn.LSTMCell(cin, cout, opx, oph, nonlinearity=torch.tanh)



if __name__ == '__main__':
    num_classes, cin, tbins, height, width = 2, 3, 100, 128, 128
    batchsize = 8
    epochs = 7
    cuda = 1
    train_iter = 100
    nclasses = 2
    dataset = SquaresVideos(t=tbins, c=cin, h=height, w=width, batchsize=batchsize, max_classes=num_classes-1, render=False)
    dataset.num_frames = train_iter

    hidden = 128
    net = rnn.RNN(nn.Sequential(fc_lstm(4, hidden), fc_lstm(hidden, hidden), nn.Linear(hidden, 4)))

    # net = Sequence(batchsize).double()
    rnn = rnn.init(net, 1000)

    if cuda:
        net.cuda()


    logdir = '/home/eperot/boxexp/fc_alpha_random/'
    writer = SummaryWriter(logdir)


    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = nn.MSELoss()

    proba = 1
    gt_every = 1
    alpha = 0.8

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

            x = encode_boxes_fc(targets) / height
            if cuda:
                x = x.cuda()

            target = x[1:]
            out = net(x, alpha=alpha)[:-1]

            loss = criterion(out, target)

            loss.backward()
            optimizer.step()
            if batch_idx%3 == 0:
                print('loss:', loss.item())
            writer.add_scalar('train_loss', loss.data.item(), batch_idx + epoch * len(dataset))
            

        #val round: make video, initialize from real seq
        print('DEMO:')

        with torch.no_grad():

            periods = 1
            nrows = 2
            ncols = batchsize / nrows
            net.reset()
            grid = np.zeros((periods * tbins, nrows, ncols, dataset.height, dataset.width, 3), dtype=np.uint8)

            for period in range(periods):
                _, targets = dataset.next()

                x = encode_boxes_fc(targets) / height
                if cuda:
                    x = x.cuda()
                out = net(x)
                for t in range(tbins):
                    for i in range(batchsize):
                        y = i / ncols
                        x = i % ncols
                        img = grid[t + period * tbins, y, x]

                        box = (out[t, i] * height).cpu().data.numpy().astype(np.int32)

                        pt1 = (box[0], box[1])
                        pt2 = (box[2], box[3])
                        cv2.rectangle(img, pt1, pt2, (0,255,0), 2)
                        grid[t + period * tbins, y, x] = img


            grid = grid.swapaxes(2, 3).reshape(periods * tbins, nrows * dataset.height, ncols * dataset.width, 3)
            utils.add_video(writer, 'test_with_xt', grid, global_step=epoch, fps=30)

            periods = 5
            grid = np.zeros((periods * tbins, nrows, ncols, dataset.height, dataset.width, 3), dtype=np.uint8)
            batch, _ = dataset.next()
            _, targets = dataset.next()

            x = encode_boxes_fc(targets) / height
            if cuda:
                x = x.cuda()

            out = net(x, future=int(periods*tbins))
            for t in range(periods*tbins):
                for i in range(batchsize):
                    y = i / ncols
                    x = i % ncols
                    img = grid[t, y, x]

                    box = (out[t, i] * height).cpu().data.numpy().astype(np.int32)
                    pt1 = (box[0], box[1])
                    pt2 = (box[2], box[3])
                    color = (0,255,0) if t <= tbins else (255,0,0)
                    cv2.rectangle(img, pt1, pt2, color, 2)

                    grid[t, y, x] = img

            grid = grid.swapaxes(2, 3).reshape(periods * tbins, nrows * dataset.height, ncols * dataset.width, 3)
            utils.add_video(writer, 'test_hallucinate', grid, global_step=epoch, fps=30)

        scheduler.step()
        proba *= 0.9
        alpha *= 0.9
        #alpha = max(0.1, alpha)








