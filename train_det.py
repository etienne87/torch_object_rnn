from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from functools import partial

from core.ssd.model import SSD
from core.trainer import DetTrainer
from core.utils import opts

from datasets.moving_mnist_detection import MovingMnistDataset
from datasets.moving_coco_detection import MovingCOCODataset


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SSD Training')
    parser.add_argument('logdir', type=str, help='where to save')
    parser.add_argument('--path', type=str, default='', help='path to dataset')
    parser.add_argument('--batchsize', type=int, default=8, help='batchsize')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate #1e-5 is advised')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--train_iter', type=int, default=500, help='#iter / train epoch')
    parser.add_argument('--test_iter', type=int, default=50, help='#iter / test epoch')
    parser.add_argument('--epochs', type=int, default=100, help='num epochs to train')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--log_every', type=int, default=10, help='log every')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--test_every', default=1, help='test_every')
    parser.add_argument('--save_every', default=10, help='save_every')
    parser.add_argument('--num_workers', default=2, help='save_every')
    return parser.parse_args()


def make_moving_mnist(args):
    # won't work with several workers
    datafunc = partial(torch.utils.data.DataLoader, batch_size=args.batchsize, num_workers=0,
                       shuffle=False, collate_fn=opts.video_collate_fn_with_reset_info, pin_memory=True)
    train_dataset = MovingMnistDataset(args.batchsize,
                                       args.time, args.height, args.width, 3, train=True)
    test_dataset = MovingMnistDataset(args.batchsize,
                                      args.time, args.height, args.width, 3, train=False)
    train_dataset.num_batches = args.train_iter
    test_dataset.num_batches = args.test_iter
    train_dataset = datafunc(train_dataset)
    test_dataset = datafunc(test_dataset)
    classes = 11
    return train_dataset, test_dataset, classes


def make_moving_coco(args):
    dataDir = '/home/etienneperot/workspace/data/coco'
    datafunc = partial(torch.utils.data.DataLoader, batch_size=args.batchsize, num_workers=args.num_workers,
                                         shuffle=True, collate_fn=opts.video_collate_fn, pin_memory=True)

    train_dataset = MovingCOCODataset(dataDir, dataType='train2017', time=args.time, height=args.height, width=args.width)
    test_dataset = MovingCOCODataset(dataDir, dataType='val2017', time=args.time, height=args.height, width=args.width)
    train_dataset = datafunc(train_dataset)
    test_dataset = datafunc(test_dataset)
    classes = len(train_dataset.dataset.catNms) + 1
    return train_dataset, test_dataset, classes


def main():
    args = parse_args()

    os.environ['OMP_NUM_THREADS'] = '1'

    classes, cin, time, height, width = 11, 3, 10, 256, 256

    args.time = time
    args.cin = cin
    args.height = height
    args.width = width

    # Dataset
    print('==> Preparing dataset..')


    train_dataset, test_dataset, classes = make_moving_mnist(args)
    # train_dataset, test_dataset, classes = make_moving_coco(time, height, width, args)



    # Model
    print('==> Building model..')
    net = SSD(num_classes=classes, cin=cin, height=height, width=width, act="softmax")

    if args.cuda:
        net.cuda()
        cudnn.benchmark = True


    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    start_epoch = 0  # start from epoch 0 or last epoch
    if args.resume:
        print('==> Resuming from checkpoint..')
        start_epoch = opts.load_last_checkpoint(net, optimizer, args.logdir)


    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min') #patience=3, verbose=True)
    trainer = DetTrainer(args.logdir, net, optimizer)


    for epoch in range(start_epoch, args.epochs):
        trainer.train(epoch, train_dataset, args)
        map = trainer.evaluate(epoch, test_dataset, args)

        trainer.writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step(map)

        if epoch%args.test_every == 0:
            trainer.test(epoch, test_dataset, args)

        if epoch%args.save_every == 0:
            trainer.save_ckpt(epoch, 'checkpoint#'+str(epoch))

if __name__ == '__main__':
    main()
