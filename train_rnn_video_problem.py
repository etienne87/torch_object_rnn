from __future__ import print_function

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import core.recurrent as rnn
import core.utils as utils
from toy_pbm_detection import SquaresVideos
import numpy as np
from tensorboardX import SummaryWriter
from functools import partial


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


def vip(inp, oup, op, stride):
    return nn.Sequential(op(inp, oup, stride=2*stride),
           nn.UpsamplingBilinear2d(scale_factor=2))


def conv_lstm(cin, cout, stride=1):
    oph = partial(nn.Conv2d, kernel_size=3, padding=1, stride=1, bias=True)
    opx = partial(conv_bn, stride=stride)
    return rnn.LSTMCell(cin, cout, opx, oph, nonlinearity=torch.tanh)



class Warping(nn.Module):
    def __init__(self, img_rows, img_cols):
        super(Warping, self).__init__()
        grid_h, grid_w = torch.meshgrid([torch.linspace(-1., 1., img_rows), torch.linspace(-1., 1., img_cols)])

        self.grid = nn.Parameter(torch.cat((grid_w[None, :, :, None],
                                            grid_h[None, :, :, None]), 3), False)

    def forward(self, img_1, disp_f):
        warp_im_1 = F.grid_sample(img_1, disp_f.permute(0, 2, 3, 1).contiguous() + self.grid)

        return warp_im_1


class RAECell(nn.Module):
    def __init__(self, inp, mode='normal'):
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



        self.lstm1 = conv_lstm(128, 128) #stateful
        self.lstm2 = conv_lstm(128, 128) #stateful

        self.mode = mode

        if mode == 'homography':
            #global homography per image
            self.avgpool = nn.AdaptiveAvgPool2d((3,3))
            self.fc_loc = nn.Sequential(
                nn.Linear(128 * 3 ** 2, 32),
                nn.ReLU(True),
                nn.Linear(32, 4)
            )
            # initialized to identity in bias + zero transformation
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0,
                                                          0, 1, 0], dtype=torch.float))
        else:    
            self.dconv3 = conv_dw(128, 64, 1)
            self.dconv2 = conv_dw(64, 32, 1)
           
            if mode == 'flow':
                self.dconv1 = nn.Conv2d(32, 2, 3, stride=1, padding=1)
                self.warper = Warping(64, 64)
            else:
                self.dconv1 = nn.Conv2d(32, inp, 3, stride=1, padding=1)
        
    def forward(self, images):
        x = images
        x, p1 = self.pool1(self.conv1(x))
        x, p2 = self.pool2(self.conv2(x))
        x, p3 = self.pool3(self.conv3(x))

        x = self.lstm1(x)
        x = self.lstm2(x) + x

        if self.mode == 'homography':
            xs = self.avgpool(x)
            xs = xs.view(xs.size(0), -1)
            theta = self.fc_loc(xs)
            theta = theta.view(-1, 2, 3)
            grid = F.affine_grid(theta, images.size())
            x = nn.functional.grid_sample(images, grid)
        else:
            x = self.dconv3(self.unpool3(x, p3))
            x = self.dconv2(self.unpool2(x, p2))
            x = self.dconv1(self.unpool1(x, p1))

            if hasattr(self, "warper"):
                x = self.warper(images, x)

        return x




if __name__ == '__main__':
    classes, cin, time, height, width = 2, 3, 10, 64, 64
    batchsize = 16
    epochs = 100
    cuda = True
    train_iter = 1000
    dataset = SquaresVideos(t=time, c=cin, h=height, w=width, batchsize=batchsize, normalize=False)
    dataset.num_frames = train_iter

    seq = rnn.RNN(RAECell(cin, mode='normal'))
    if cuda:
        seq.cuda()

    logdir = '/home/eperot/logdir/rnn_cv_combine_peephole/'
    writer = SummaryWriter(logdir)


    optimizer = optim.Adam(seq.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()

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

            batch, _ = dataset.next()
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

        #val round: make video, initialize from real seq
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


        scheduler.step()





