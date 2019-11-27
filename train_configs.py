from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.moving_mnist_detection import make_moving_mnist
from datasets.coco_detection import make_still_coco
from core.single_stage_detector import SingleStageDetector

def mnist_rnn(args):
    args.lr = 1e-3
    args.wd = 1e-4
    args.epochs = 20
    print('==> Preparing dataset..')
    train, val, num_classes = make_moving_mnist(args)
    print('==> Building model..')
    net = SingleStageDetector.tiny_rnn_fpn(3, num_classes, act='sigmoid', loss='_focal_loss')
    return net, train, val

def coco_resnet(args):
    args.lr = 1e-5
    args.wd = 1e-4
    print('==> Preparing dataset..')
    train, val, num_classes = make_still_coco(args.path, args.batchsize, args.num_workers)
    print('==> Building model..')
    net = SingleStageDetector.resnet50_fpn(3, num_classes, act="sigmoid", loss='_focal_loss', nlayers=3)
    return net, train, val