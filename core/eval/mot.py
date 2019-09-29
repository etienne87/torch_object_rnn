from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import motmetrics as mm

from datasets.moving_box_detection import SquaresVideos
import cv2

def deform_gt(gt, r=1./16):
    x1, y1, x2, y2, c = np.split(gt, 5, axis=1)
    w, h = x2-x1, y2-y1
    s = np.minimum(w, h) / 16.0
    cx = x1 + w/2 + np.random.uniform(-s, s)
    cy = y1 + h/2 + np.random.uniform(-s, s)
    f = np.random.uniform(1 - r, 1 + r)
    w, h = w * f, h * f
    det =  np.hstack([cx-w/2, cy-h/2, cx+w/2, cy+h/2, c]).astype(np.int32)
    return det

def xyxy2xywh(boxes):
    xy1 = boxes[:,:2]
    xy2 = boxes[:,2:]
    wh = xy2-xy1
    return np.hstack([xy1, wh])


def show(gt, dt, height, width):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for j in range(gt.shape[0]):
        cv2.rectangle(img, (gt[j, 0], gt[j, 2]), (gt[j, 1], gt[j, 3]), (255, 0, 0), 1)
    for k in range(dt.shape[0]):
        cv2.rectangle(img, (dt[k, 0], dt[k, 2]), (dt[k, 1], dt[k, 3]), (0, 0, 255), 1)
    cv2.imshow('img', img)
    cv2.waitKey()


if __name__ == '__main__':
    dataloader = SquaresVideos(t=30, c=1, h=256, w=256,  max_objects=5, batchsize=1, render=False)
    acc = mm.MOTAccumulator(auto_id=True)


    a = np.array([
        [0, 0, 20, 100],  # Format X, Y, Width, Height
        [0, 0, 0.8, 1.5],
    ])

    b = np.array([
        [0, 0, 1, 2],
        [0, 0, 1, 1],
        [0.1, 0.2, 2, 2],
    ])


    ious = mm.distances.iou_matrix(a, b, max_iou=0.5)

    acc.update([1, 2], [1, 2, 3], ious)

    # for i in range(10):
    #     _, boxes_tn = dataloader.next()
    #
    #     for boxes in boxes_tn:
    #         gt = boxes[0].numpy()
    #         dt = deform_gt(gt)
    #         # show(gt, dt, 256, 256)
    #
    #         gt = xyxy2xywh(gt[:,:4])
    #         dt = xyxy2xywh(dt[:,:4])
    #
    #         ious = mm.distances.iou_matrix(gt, dt, max_iou=0.5)
    #         acc.update(gt, dt, ious)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc')

    print(summary)

