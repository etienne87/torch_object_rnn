from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np

import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from core.utils import image as imutil
from core.utils import opts
from core.utils import vis
import cv2
from functools import partial




def resize_image_and_boxes(image, boxes, dsize):
    height, width = image.shape[:2]
    img = cv2.resize(image, dsize, 0, 0, cv2.INTER_LINEAR)
    ry, rx = dsize[1]/height, dsize[0]/width
    boxes[:, [0, 2]] *= rx
    boxes[:, [1, 3]] *= ry
    return img, boxes


def clamp_boxes(boxes, height, width):
    boxes[:, [0, 2]] = np.maximum(np.minimum(boxes[:, [0, 2]], width), 0)
    boxes[:, [1, 3]] = np.maximum(np.minimum(boxes[:, [1, 3]], height), 0)
    return boxes


def discard_too_small(boxes, min_size=30):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    ids = np.where((w > min_size)*(h > min_size))
    return boxes[ids]


def wrap_boxes(boxes, height, width):
    allbox = [boxes]
    for y in [-height, 0, height]:
        for x in [-width, 0, width]:
            if x == 0 and y == 0:
                continue
            shift = boxes.copy()
            shift[:, [0, 2]] += x
            shift[:, [1, 3]] += y
            allbox.append(shift)
    return np.concatenate(allbox, 0)


def viz_batch(seq_im, seq_boxes):
    video = seq_im.permute([0, 2, 3, 1]).cpu().numpy() * 127 + 127
    video = np.split(video, 10)
    seq_boxes = [item.cpu().numpy().copy().astype(np.int32) for item in seq_boxes]
    for im, boxes in zip(video, seq_boxes):
        im = im[0].astype(np.uint8).copy()
        bboxes = vis.boxarray_to_boxes(boxes, boxes[:, -1], mcoco.catNms)
        img_ann = vis.draw_bboxes(im, bboxes, 2, colormap=cv2.COLORMAP_JET)
        cv2.imshow('im', img_ann)
        cv2.waitKey(5)


class MovingCOCODataset(Dataset):
    def __init__(self, dataDir, dataType, catNms=None, time=10, height=480, width=640):
        self.dataDir = dataDir
        self.dataType = dataType
        self.annFile = '{}/annotations/instances_{}.json'.format(self.dataDir, self.dataType)
        self.coco = COCO(self.annFile)
        self.catNms = ['person',
                       'car',
                       'motorbike',
                       'truck',
                       'bus',
                       'bicycle'] if catNms is None else catNms
        self.catIds = self.coco.getCatIds(catNms=self.catNms)
        self.imgIds = self._getImgIdsUnion(self.catIds)
        self.time = time
        self.height = height
        self.width = width
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.catIds)}
        self.border_mode = cv2.BORDER_WRAP
        self.img_warp = partial(cv2.warpPerspective, dsize=(width, height), borderMode=self.border_mode)

    def _getImgIdsUnion(self, catIds):
        out = []
        for catId in catIds:
            out += self.coco.getImgIds(catIds=[catId])
        out = set(out)
        return list(out)

    def get_boxes(self, anns):
        boxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            label = self.cat2label.get(ann['category_id'], -1)
            if label == -1:
                continue
            boxes.append(np.array([[x, y, x + w, y + h, label]], dtype=np.float32))
        boxes = np.concatenate(boxes, axis=0)
        return boxes

    def __getitem__(self, item):
        img = self.coco.loadImgs(self.imgIds[item])[0]
        file_name = os.path.join(self.dataDir, 'images', self.dataType, img['file_name'])
        image = cv2.imread(file_name)
        annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        boxes = self.get_boxes(anns)

        image, boxes = resize_image_and_boxes(image, boxes, (self.width, self.height))
        imseq, boxseq = self.trip(image, boxes, self.time)
        return imseq, boxseq

    def __len__(self):
        return len(self.imgIds)

    def trip(self, image, boxes, tbins):
        height, width = image.shape[:2]
        voyage = imutil.PlanarVoyage(height, width)
        if self.border_mode == cv2.BORDER_WRAP:
            boxes = wrap_boxes(boxes, height, width)
        imseq = []
        boxseq = []

        for _ in range(tbins):
            G_0to2 = voyage()
            out = self.img_warp(image, G_0to2, flags=cv2.INTER_LINEAR).astype(np.float32)/128 - 1
            labels = boxes[:, 4:5]
            tboxes = imutil.cv2_apply_transform_boxes(boxes[:, :4], G_0to2)
            tboxes = clamp_boxes(tboxes, height, width)
            tboxes = np.concatenate([tboxes, labels], 1)
            tboxes = discard_too_small(tboxes, 10)
            out = torch.from_numpy(out).permute([2, 0, 1])
            tboxes = torch.from_numpy(tboxes)
            imseq.append(out[None])
            boxseq.append(tboxes)
        imseq = torch.cat(imseq, 0)

        return imseq, boxseq


if __name__ == '__main__':
    dataDir = '/home/etienneperot/workspace/data/coco'
    dataType = 'val2017'

    mcoco = MovingCOCODataset(dataDir, dataType, time=10)
    loader = torch.utils.data.DataLoader(mcoco, batch_size=10, num_workers=3,
                           shuffle=True, collate_fn=opts.video_collate_fn, pin_memory=True)

    # while 1:
    #     seq_im, seq_boxes = mcoco[np.random.randint(0, len(mcoco))]
    #     viz_batch(seq_im, seq_boxes)

    start = time.time()
    for x, y in loader:

        end = time.time()
        print(end - start, ' time loading')
        for i in range(10):
            video = x[:, i]
            seq_boxes = [item[i] for item in y]
            viz_batch(video, seq_boxes)

        start = time.time()
        print(start-end, ' time showing')