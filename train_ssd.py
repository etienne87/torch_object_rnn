from __future__ import print_function

import os
import time as timing
import argparse
import numpy as np
import cv2

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from ssd_v2 import SSD
from box_coder import SSDBoxCoder
from ssd_loss import SSDLoss
from trainer import SSDTrainer

from toy_pbm_detection import SquaresVideos
from utils import draw_bboxes, make_single_channel_display, filter_outliers


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SSD Training')
    parser.add_argument('--path', type=str, default='', help='path to dataset')
    parser.add_argument('--batchsize', type=int, default=32, help='batchsize')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--train_iter', type=int, default=200, help='#iter / train epoch')
    parser.add_argument('--test_iter', type=int, default=200, help='#iter / test epoch')
    parser.add_argument('--epochs', type=int, default=1000, help='num epochs to train')
    parser.add_argument('--model', default='./checkpoints/ssd300_v2.pth', type=str, help='initialized model path')
    parser.add_argument('--checkpoint', default='./checkpoints/ckpt.pth', type=str, help='checkpoint path')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--log_every', type=int, default=10, help='log every')
    return parser.parse_args()


def main():
    args = parse_args()

    classes, cin, time, height, width = 2, 1, 5, 128, 128

    nrows = 4

    # Dataset
    print('==> Preparing dataset..')
    dataset = SquaresVideos(t=time, c=cin, h=height, w=width, batchsize=args.batchsize, normalize=False, cuda=args.cuda)


    # Model
    print('==> Building model..')
    net = SSD(num_classes=classes, cin=cin, height=height, width=width)

    if args.cuda:
        net.cuda()
        cudnn.benchmark = True

    best_loss = float('inf')  # best test loss
    start_epoch = 0  # start from epoch 0 or last epoch
    if args.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']

    box_coder = SSDBoxCoder(net)

    if args.cuda:
        box_coder.cuda()

    img_size = (dataset.height, dataset.width)

    criterion = SSDLoss(num_classes=classes)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    trainer = SSDTrainer(net, box_coder, criterion, optimizer)

    for epoch in range(start_epoch, start_epoch + 200):
        trainer.train(epoch, dataset, args)
        trainer.test(epoch, dataset, nrows, args)
        # scheduler.step()
        # trainer.save_ckpt(epoch)



if __name__ == '__main__':
    main()