from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from functools import partial

# from core.ssd.model import SSD
from core.single_stage_detector import SingleStageDetector
from core.trainer import DetTrainer
from core.utils import opts

from datasets.moving_mnist_detection import MovingMnistDataset
from datasets.moving_coco_detection import MovingCOCODataset
from datasets.coco_detection import make_coco_dataset as make_still_coco


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SSD Training')
    parser.add_argument('logdir', type=str, help='where to save')
    parser.add_argument('--path', type=str, default='', help='path to dataset')
    parser.add_argument('--batchsize', type=int, default=8, help='batchsize')
    parser.add_argument('--time', type=int, default=8, help='timesteps')

    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate #1e-5 is advised')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--train_iter', type=int, default=500, help='#iter / train epoch')
    parser.add_argument('--test_iter', type=int, default=50, help='#iter / test epoch')
    parser.add_argument('--epochs', type=int, default=100, help='num epochs to train')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--log_every', type=int, default=10, help='log every')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--test_every', type=int, default=1, help='test_every')
    parser.add_argument('--save_every', type=int, default=2, help='save_every')
    parser.add_argument('--num_workers', type=int, default=2, help='save_every')
    parser.add_argument('--just_test', action='store_true')
    parser.add_argument('--just_val', action='store_true')
    return parser.parse_args()


def make_moving_mnist(args):
    # won't work with several workers
    datafunc = partial(torch.utils.data.DataLoader, batch_size=args.batchsize, num_workers=0,
                       shuffle=False, collate_fn=opts.video_collate_fn_with_reset_info, pin_memory=True)
    train_dataset = MovingMnistDataset(args.batchsize,
                                       args.time, args.height, args.width, 3, train=True, max_objects=5)
    test_dataset = MovingMnistDataset(args.batchsize,
                                      args.time, args.height, args.width, 3, train=False, max_objects=5)
    train_dataset.num_batches = args.train_iter
    test_dataset.num_batches = args.test_iter
    train_dataset = datafunc(train_dataset)
    test_dataset = datafunc(test_dataset)
    classes = 10
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

    cin, height, width = 3, 256, 256

    args.cin = 3
    args.height = 256
    args.width = 256

    coco_path = '/home/etienneperot/workspace/data/coco/'
    coco_path = '/home/prophesee/work/etienne/datasets/coco/'
    # Dataset
    print('==> Preparing dataset..')


    # train_dataset, test_dataset, classes = make_moving_mnist(args)
    # train_dataset, test_dataset, classes = make_moving_coco(time, height, width, args)
    train_dataset, test_dataset, classes = make_still_coco(coco_path, args.batchsize, args.num_workers)

    args.is_video_dataset = False

    print('classes: ', classes)
    # Model
    print('==> Building model..')
    net = SingleStageDetector.mobilenet_v2_fpn(cin, classes, act="softmax")

    if args.cuda:
        net.cuda()
        cudnn.benchmark = True


    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    start_epoch = 0  # start from epoch 0 or last epoch
    if args.resume:
        print('==> Resuming from checkpoint..')
        start_epoch = opts.load_last_checkpoint(args.logdir, net, optimizer) + 1


    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min') 
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    trainer = DetTrainer(args.logdir, net, optimizer)

    if args.just_test: 
        trainer.test(start_epoch + 1, test_dataset, args)
        exit()
    elif args.just_val:
        mean_ap_50 = trainer.evaluate(start_epoch + 1, test_dataset, args)
        print('mean_ap_50: ', mean_ap_50)
        exit()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = trainer.train(epoch, train_dataset, args)
        mean_ap_50 = trainer.evaluate(epoch, test_dataset, args)

        trainer.writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step(mean_ap_50)
        # scheduler.step(np.mean(epoch_loss))

        # if epoch%args.test_every == 0:
        #     trainer.test(epoch, test_dataset, args)

        if epoch%args.save_every == 0:
            trainer.save_ckpt(epoch, 'checkpoint#'+str(epoch))

if __name__ == '__main__':
    main()
