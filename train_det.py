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


from core.single_stage_detector import SingleStageDetector
from core.trainer import DetTrainer
from core.utils import opts

from datasets.moving_mnist_detection import make_moving_mnist
from datasets.coco_detection import make_still_coco


try:
    from apex import amp
except ImportError:
    print('WARNING apex not installed, half precision will not be available')


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
    
    parser.add_argument('--log_every', type=int, default=10, help='log every')
    
    parser.add_argument('--test_every', type=int, default=1, help='test_every')
    parser.add_argument('--save_every', type=int, default=2, help='save_every')
    parser.add_argument('--num_workers', type=int, default=2, help='save_every')
    parser.add_argument('--just_test', action='store_true')
    parser.add_argument('--just_val', action='store_true')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--half', action='store_true', help='use fp16')
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ['OMP_NUM_THREADS'] = '1'

    cin, height, width = 3, 256, 256

    args.cin = 3
    args.height = 256
    args.width = 256

    # Dataset
    print('==> Preparing dataset..')


    # train_dataset, test_dataset, classes = make_moving_mnist(args)
    train_dataset, test_dataset, classes = make_still_coco(args.path, args.batchsize, args.num_workers)

    args.is_video_dataset = False

    print('classes: ', classes)
    # Model
    print('==> Building model..')
    # net = SingleStageDetector.mobilenet_v2_fpn(cin, classes, act="softmax")
    # net = SingleStageDetector.resnet50_fpn(cin, classes, act="softmax", loss='_ohem_loss', nlayers=0)
    net = SingleStageDetector.resnet50_fpn(cin, classes, act="softmax", loss='_ohem_loss', nlayers=3)
    

    if args.cuda:
        net.cuda()
        cudnn.benchmark = True


    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

    if args.half:
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1", verbosity=0)
        print('FP16 activated')

    start_epoch = 0  # start from epoch 0 or last epoch
    if args.resume:
        print('==> Resuming from checkpoint..')
        start_epoch = opts.load_last_checkpoint(args.logdir, net, optimizer) + 1

    print('Current learning rate: ', optimizer.param_groups[0]['lr'])

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

        if epoch%args.save_every == 0:
            trainer.save_ckpt(epoch, 'checkpoint#'+str(epoch))

        # mean_ap_50 = trainer.evaluate(epoch, test_dataset, args)

        trainer.writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        # scheduler.step(mean_ap_50)
        scheduler.step(np.mean(epoch_loss))

        # if epoch%args.test_every == 0:
        #     trainer.test(epoch, test_dataset, args)


if __name__ == '__main__':
    main()
