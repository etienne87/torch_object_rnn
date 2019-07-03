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



if __name__ == '__main__':
    num_classes, cin, tbins, height, width = 2, 3, 200, 64, 64
    batchsize = 8
    epochs = 100
    cuda = 1
    train_iter = 100
    nclasses = 2
    dataset = SquaresVideos(t=tbins, c=cin, h=height, w=width, batchsize=batchsize, max_classes=num_classes-1)
    dataset.num_frames = train_iter

    net = rnn.RNN(nn.Sequential(rnn.lstm_fc(4, 128), rnn.lstm_fc(128, 128), nn.Linear(128, 4)))

    if cuda:
        net.cuda()


    logdir = '/home/eperot/boxexp/tmp_fc/'
    writer = SummaryWriter(logdir)


    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = nn.MSELoss()

    proba = 1
    gt_every = 1
    alpha = 1

    for epoch in range(epochs):
        #train round
        print('TRAIN: PROBA RESET: ', proba, ' ALPHA: ', alpha)
        net.reset()
        for batch_idx, data in enumerate(dataset):
            # if np.random.rand() < proba:
            #     dataset.reset()
            #     net.reset()
            dataset.reset()
            net.reset()

            _, targets = dataset.next()
            optimizer.zero_grad()

            x = encode_boxes_fc(targets) / 64
            if cuda:
                x = x.cuda()

            target = x[1:]
            out = net(x[:-1], alpha=alpha)#[:-1]

            loss = criterion(out, target)

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
                x = encode_boxes_fc(targets) / 64
                if cuda:
                    x = x.cuda()
                out = net(x)
                for t in range(tbins):
                    for i in range(batchsize):
                        y = i / ncols
                        x = i % ncols
                        img = grid[t + period * tbins, y, x]

                        box = (out[t, i] * 64).cpu().data.numpy()

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

            x = encode_boxes_fc(targets) / 64
            if cuda:
                x = x.cuda()

            out = net(x, future=int(periods*tbins))
            for t in range(periods*tbins):
                for i in range(batchsize):
                    y = i / ncols
                    x = i % ncols
                    img = grid[t, y, x]

                    box = (out[t, i] * 64).cpu().data.numpy()
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
        alpha = max(0.1, alpha)








