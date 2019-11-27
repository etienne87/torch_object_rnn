from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import torch

from core.trainer import DetTrainer

import train_configs as cfg 


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SSD Training')
    parser.add_argument('logdir', type=str, help='where to save')

    parser.add_argument('--config', type=str, default='mnist_rnn')

    parser.add_argument('--path', type=str, default='', help='path to dataset')
    parser.add_argument('--batchsize', type=int, default=8, help='batchsize')
    parser.add_argument('--time', type=int, default=8, help='timesteps')

    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate #1e-5 is advised')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay #1e-4 advised')
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
    parser.add_argument('--clip_grad_norm', action='store_true', help='clip gradient')
    return parser.parse_args()


def main():
    args = parse_args()

    net, optimizer, scheduler, train_dataset, test_dataset, start_epoch = getattr(cfg, args.config)(args)

    trainer = DetTrainer(args.logdir, net, optimizer, scheduler)

    if args.just_test: 
        trainer.test(start_epoch + 1, test_dataset, args)
        exit()
    elif args.just_val:
        mean_ap_50 = trainer.evaluate(start_epoch + 1, test_dataset, args)
        print('mean_ap_50: ', mean_ap_50)
        exit()

    for epoch in range(start_epoch, args.epochs):
        trainer.train(epoch, train_dataset, args)

        if epoch%args.save_every == 0:
            trainer.save_ckpt(epoch, 'checkpoint#'+str(epoch))

        mean_ap_50 = trainer.evaluate(epoch, test_dataset, args)

        trainer.writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step(mean_ap_50)
      
        # if epoch%args.test_every == 0:
        #     trainer.test(epoch, test_dataset, args)


if __name__ == '__main__':
    main()
