from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from core.ssd.model import SSD
from core.ctdet.model import CenterNet
from core.trainer import DetTrainer
from core.utils import opts

from data.moving_mnist_detection import MovingMnistDataset


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SSD Training')
    parser.add_argument('logdir', type=str, help='where to save')
    parser.add_argument('--path', type=str, default='', help='path to dataset')
    parser.add_argument('--batchsize', type=int, default=8, help='batchsize')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--train_iter', type=int, default=100, help='#iter / train epoch')
    parser.add_argument('--test_iter', type=int, default=10, help='#iter / test epoch')
    parser.add_argument('--epochs', type=int, default=1000, help='num epochs to train')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--log_every', type=int, default=10, help='log every')
    parser.add_argument('--save_video', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ['OMP_NUM_THREADS'] = '1'

    classes, cin, time, height, width = 11, 3, 10, 128, 128
    nrows = 4

    # Dataset
    print('==> Preparing dataset..')

    train_dataset = MovingMnistDataset(args.batchsize, time, height, width, cin, train=True)
    test_dataset = MovingMnistDataset(args.batchsize, time, height, width, cin, train=False)

    train_dataset.num_frames = args.train_iter
    dataloader = train_dataset


    # Model
    print('==> Building model..')
    # net = SSD(num_classes=classes, cin=cin, height=height, width=width, act="softmax")
    net = CenterNet(num_classes=classes, cin=cin, height=height, width=width)

    if args.cuda:
        net.cuda()
        net.box_coder.cuda()
        net.criterion.cuda()
        cudnn.benchmark = True


    start_epoch = 0  # start from epoch 0 or last epoch
    if args.resume:
        print('==> Resuming from checkpoint..')
        start_epoch = opts.load_last_checkpoint(net, args.logdir)


    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    trainer = DetTrainer(args.logdir, net, optimizer)

    for epoch in range(start_epoch, args.epochs):
        trainer.train(epoch, dataloader, args)
        # trainer.validate(epoch, test_dataset, args)
        trainer.test(epoch, test_dataset, nrows, args)
        trainer.save_ckpt(epoch, args)
        trainer.writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step()

if __name__ == '__main__':
    main()
