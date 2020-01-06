from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import cv2

from core.backbones import Vanilla, FPN
from core.rpn import BoxHead
from core.single_stage_detector import SingleStageDetector


def draw_anchors(anchors, height=256, width=256):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    anchors = anchors.cpu().data.numpy()

    img[...] = 0
    for i in range(len(anchors)):
        anchor = anchors[i]
        pt1, pt2 = (anchor[0], anchor[1]), (anchor[2], anchor[3])

        cv2.rectangle(img, pt1, pt2, (255, 0, 0), 1)
        cv2.imshow('img', img)
        cv2.waitKey(2)


imgsize = 256

net = SingleStageDetector(Vanilla, BoxHead, 3, 10, 'softmax',
                          loss='_ohem_loss',
                          ratios=[1.0],
                          scales=[1.0])

x = torch.rand(1, 1, 3, imgsize, imgsize)
xs = net.feature_extractor(x)

print([item.shape[-2:] for item in xs])

anchors, anchors_xyxy = net.box_coder(xs, x.shape[-2:])

draw_anchors(anchors_xyxy, height=imgsize*2, width=imgsize*2)