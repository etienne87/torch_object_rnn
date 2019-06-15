from __future__ import print_function

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import core.recurrent as rnn
import core.utils as utils
from toy_pbm_detection import SquaresVideos
import numpy as np
from tensorboardX import SummaryWriter


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



class RAECell(nn.Module):
    def __init__(self, inp):
        super(RAECell, self).__init__()
        self.conv1 = conv_bn(inp, 32, 1)
        self.conv2 = conv_dw(32, 64, 1)
        self.conv3 = conv_dw(64, 128, 1)

        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.unpool3 = nn.MaxUnpool2d(2, 2)

        self.lstm = rnn.lstm_conv(128, 128)

        self.dconv3 = conv_dw(128, 64, 1)
        self.dconv2 = conv_dw(64, 32, 1)
        self.dconv1 = nn.Conv2d(32, inp, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x, p1 = self.pool1(self.conv1(x))
        x, p2 = self.pool2(self.conv2(x))
        x, p3 = self.pool3(self.conv3(x))

        x = self.lstm(x)

        x = self.dconv3(self.unpool3(x, p3))
        x = self.dconv2(self.unpool2(x, p2))
        x = self.dconv1(self.unpool1(x, p1))

        return x




if __name__ == '__main__':
    classes, cin, time, height, width = 2, 3, 10, 64, 64
    batchsize = 16
    epochs = 100
    cuda = True
    train_iter = 1000
    dataset = SquaresVideos(t=time, c=cin, h=height, w=width, batchsize=batchsize, normalize=False)
    dataset.num_frames = train_iter

    seq = rnn.RNN(RAECell(cin))
    if cuda:
        seq.cuda()

    logdir = 'logdir/rnn/'
    writer = SummaryWriter(logdir)


    optimizer = optim.Adam(seq.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()

    for epoch in range(epochs):
        #train round
        print('TRAIN:')
        seq.reset()
        for batch_idx, data in enumerate(dataset):
            batch, _ = dataset.next()
            batch = batch.cuda() if cuda else batch
            input, target = batch[:-1], batch[1:]
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            print('loss:', loss.item())
            writer.add_scalar('train_loss', loss.data.item(), batch_idx + epoch * len(dataset))

        #val round: make video, initialize from real seq
        print('DEMO:')
        with torch.no_grad():
            periods = 100
            nrows = 2
            ncols = batchsize / nrows
            grid = np.zeros((periods * time, nrows, ncols, dataset.height, dataset.width, 3), dtype=np.uint8)
            seq.reset()
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
            utils.add_video(writer, 'test', grid, global_step=epoch, fps=30)


        scheduler.step()





