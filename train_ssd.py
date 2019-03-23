from __future__ import print_function
import argparse
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from detection_module.ssd import SSD
from detection_module.box_coder import SSDBoxCoder
from detection_module.ssd_loss import SSDLoss
from detection_module.trainer import SSDTrainer
from detection_module.networks import ConvRNNFeatureExtractor, FPNRNNFeatureExtractor
from detection_module.utils import StreamDataset
from toy_pbm_detection import SquaresVideos

from detection_module.modules import ConvLSTMCell

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SSD Training')
    parser.add_argument('--path', type=str, default='', help='path to dataset')
    parser.add_argument('--batchsize', type=int, default=8, help='batchsize')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--train_iter', type=int, default=200, help='#iter / train epoch')
    parser.add_argument('--test_iter', type=int, default=200, help='#iter / test epoch')
    parser.add_argument('--epochs', type=int, default=1000, help='num epochs to train')
    parser.add_argument('--checkpoint', default='./checkpoints/', type=str, help='checkpoint path')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--log_every', type=int, default=10, help='log every')
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ['OMP_NUM_THREADS'] = '1'

    classes, cin, time, height, width = 2, 1, 5, 128, 128

    nrows = 4

    # Dataset
    print('==> Preparing dataset..')
    dataset = SquaresVideos(t=time, c=cin, h=height, w=width, batchsize=args.batchsize, normalize=False, cuda=args.cuda)

    test_dataset = SquaresVideos(t=time, c=cin, h=height, w=width, max_stops=300,
                                 batchsize=args.batchsize, normalize=False, cuda=args.cuda)
    dataset.num_frames = args.train_iter

    dataloader = StreamDataset(dataset, args.train_iter)
    test_dataloader = StreamDataset(test_dataset, args.test_iter)
    #dataloader = dataset
    # Model
    print('==> Building model..')
    net = SSD(feature_extractor=ConvRNNFeatureExtractor, num_classes=classes, cin=cin, height=height, width=width)

    if args.cuda:
        net.cuda()
        cudnn.benchmark = True

    best_loss = float('inf')  # best test loss
    start_epoch = 0  # start from epoch 0 or last epoch
    if args.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']

    box_coder = SSDBoxCoder(net)

    if args.cuda:
        box_coder.cuda()


    criterion = SSDLoss(num_classes=classes)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    trainer = SSDTrainer(net, box_coder, criterion, optimizer)

    for epoch in range(start_epoch, args.epochs):
        trainer.train(epoch, dataloader, args)
        trainer.val(epoch, test_dataloader, args)
        trainer.test(epoch, test_dataset, nrows, args)
        trainer.save_ckpt(epoch, args)
        trainer.writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step()

        for module in net.extractor.modules():
            if isinstance(module, ConvLSTMCell):
                print('alpha: ', module.alpha)

if __name__ == '__main__':
    main()