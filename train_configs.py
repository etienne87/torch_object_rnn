from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch.optim as optim
import torch.backends.cudnn as cudnn

from datasets.moving_mnist_detection import make_moving_mnist
from datasets.coco_detection import make_still_coco
from core.single_stage_detector import SingleStageDetector
from core.utils import opts

try:
    from apex import amp
except ImportError:
    print('WARNING apex not installed, half precision will not be available')


def adam_optim(net, args):
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.cuda:
        net.cuda()
        cudnn.benchmark = True

    if args.half:
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1", verbosity=0)
        print('FP16 activated')
    start_epoch = 0  # start from epoch 0 or last epoch
    if args.resume:
        print('==> Resuming from checkpoint..')
        start_epoch = opts.load_last_checkpoint(args.logdir, net, optimizer) + 1
    print('Current learning rate: ', optimizer.param_groups[0]['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min') 
    return net, optimizer, scheduler, start_epoch


def mnist_rnn(args):
    args.lr = 1e-3
    args.wd = 1e-6
    args.cin = 3
    args.height = 256
    args.width = 256
    args.epochs = 20
    print('==> Preparing dataset..')
    train, val, num_classes = make_moving_mnist(args)
    print('==> Building model..')
    net = SingleStageDetector.tiny_rnn_fpn(3, num_classes, act='sigmoid', loss='_focal_loss')
    net, optimizer, scheduler, start_epoch = adam_optim(net, args)
    return net, optimizer, scheduler, train, val, start_epoch


def coco_resnet(args):
    args.lr = 1e-5
    args.wd = 1e-4
    print('==> Preparing dataset..')
    train, val, num_classes = make_still_coco(args.path, args.batchsize, args.num_workers)
    print('==> Building model..')
    net = SingleStageDetector.resnet50_fpn(3, num_classes, act="sigmoid", loss='_focal_loss', nlayers=3)
    net, optimizer, scheduler, start_epoch = adam_optim(net, args)
    return net, optimizer, scheduler, train, val, start_epoch