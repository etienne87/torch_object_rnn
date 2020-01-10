from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import cv2

from core.backbones import Vanilla, FPN, MobileNetSSD
from core.utils import box
from core.rpn import BoxHead, SSDHead
from core.single_stage_detector import SingleStageDetector


def draw_anchors_one_level(img, anchors, color=(255,0,0), erase_every=6):    
    anchors = anchors.cpu().data.numpy()
    for i in range(len(anchors)):
        anchor = anchors[i]
        pt1, pt2 = (anchor[0], anchor[1]), (anchor[2], anchor[3])
        cv2.rectangle(img, pt1, pt2, color, 1)
        if i%erase_every == 0: 
            img[...] = 0
        elif i%erase_every == (erase_every-1):
            cv2.imshow('img', img)
            cv2.waitKey()


def draw_anchors(anchors, height=256, width=256, num_anchors=6, step=6):
    colors = [np.random.randint(0, 255, (3,)).tolist() for i in range(num_anchors)]
    img = np.zeros((height, width, 3), dtype=np.uint8)
    anchors = anchors.cpu().data.numpy()

    anchors = anchors.reshape(len(anchors) // num_anchors, num_anchors, 4)
    anchors = anchors[::step]
    anchors = anchors.reshape(-1, 4)
    img[...] = 0
    for i in range(len(anchors)):
        anchor = anchors[i]
        pt1, pt2 = (anchor[0], anchor[1]), (anchor[2], anchor[3])
        cv2.rectangle(img, pt1, pt2, colors[i % num_anchors], 2)

        if i % num_anchors == 0:
            img[...] = 0
        elif i % num_anchors == num_anchors - 1:
            cv2.imshow('img', img)
            cv2.waitKey(0)


imgsize = 512

net = SingleStageDetector(MobileNetSSD, SSDHead, 3, 10, 'softmax',
                          loss='_ohem_loss',
                          ratios=[0.5,1.0,2.0],
                          scales=[1.0, 2**(1./3), 2**(2./3)])

x = torch.rand(1, 1, 3, imgsize, imgsize)
xs = net.feature_extractor(x)

print([item.shape[-2:] for item in xs])

img = np.zeros((imgsize, imgsize, 3), dtype=np.uint8)

anchors, anchors_xyxy = net.box_coder(xs, x.shape[-2:])
draw_anchors(anchors_xyxy, height=imgsize, width=imgsize, num_anchors=net.box_coder.num_anchors, step=10)


#show level by level
# shapes = [item.shape for item in xs]
# strides = [int(imgsize / shape[-1]) for shape in shapes]
# if net.box_coder.anchors is None or shapes != net.box_coder.last_shapes:
#     default_boxes = []
#     for feature_map, anchor_layer, stride in zip(xs, net.box_coder.anchor_generators, strides):
#         anchors = anchor_layer(feature_map, stride)
#         anchors_xyxy = box.change_box_order(anchors, "xywh2xyxy")
#         draw_anchors(img, anchors_xyxy, color=np.random.randint(0, 255, (3,)).tolist(), erase_every=len(net.box_coder.scales)*len(net.box_coder.ratios)) 
#     cv2.waitKey()