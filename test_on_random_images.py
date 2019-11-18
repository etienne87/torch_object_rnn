from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import json
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from core.single_stage_detector import SingleStageDetector
from core.utils import opts, vis
from datasets.coco_detection import Normalizer
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SSD Training')
    parser.add_argument('logdir', type=str, help='where to save')
    parser.add_argument('image', type=str, default='', help='path to image')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    return parser.parse_args()


def xyxytoxywh(bbox):
    x1, y1, x2, y2 = bbox.tolist()
    w, h = (x2-x1), (y2-y1)
    return [x1, y1, w, h]

def main():
    #just for labelmap
    from datasets.coco_detection import make_coco_dataset as make_still_coco
    coco_path = '/home/prophesee/work/etienne/datasets/coco/'
    _, dataloader, classes = make_still_coco(coco_path, 1, 0)
    labelmap = dataloader.dataset.labelmap


    args = parse_args()
    
    normalizer = Normalizer()
    image = cv2.imread(args.image)
    image_normed = (image.astype(np.float32)/255.0 - normalizer.mean) / normalizer.std
    image_normed = vis.swap_channels(image_normed)
    inputs = torch.from_numpy(image_normed[None,None]).float()

    
    net = SingleStageDetector.mobilenet_v2_fpn(3, classes, act="sigmoid")
    if args.cuda:
        net.cuda()
        cudnn.benchmark = True
        inputs = inputs.cuda()
    net.eval()

    print('==> Resuming from checkpoint..')
    opts.load_last_checkpoint(args.logdir, net)
    
    boxes, labels, scores = net.get_boxes(inputs, score_thresh=0.3)[0][0]
    boxes, labels, scores = boxes.data.numpy(), labels.data.numpy(), scores.data.numpy()

    vis.draw_det_boxes(image, boxes, labels,
        labelmap,
        None,
        thickness=1,
        colormap=cv2.COLORMAP_HSV,
        colordefault=None)

    cv2.imshow('img', image)
    key = cv2.waitKey(0)

    print(inputs.shape)


if __name__ == '__main__':
    main()
