from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from torch.utils.data import Dataset
from pycocotools.coco import COCO

from core.utils import image as imutil
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


def boxarray_to_boxes(boxes):
    bboxes = []
    for i, box in enumerate(boxes):
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        bb = ('', i, pt1, pt2, None, None, None)
        bboxes.append(bb)
    return bboxes



class MovingCOCODataset(Dataset):
    def __init__(self, annFile, catNms=None, time=10, height=480, width=640):
        self.coco = COCO(annFile)
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
        self.img_warp = partial(cv2.warpPerspective, dsize=(width, height), borderMode=cv2.BORDER_WRAP)

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
            boxes.append(np.array([[x, y, x + w, y + h, label]]))
        boxes = np.concatenate(boxes, axis=0)
        return boxes

    def __getitem__(self, item):
        img = self.coco.loadImgs(self.imgIds[item])[0]
        file_name = os.path.join(dataDir, 'images', dataType, img['file_name'])
        image = cv2.imread(file_name)
        annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        boxes = self.get_boxes(anns)

        imseq, boxseq = self.trip(image, boxes, self.time)
        return imseq

        # img, boxes = resize_image_and_boxes(image, boxes, (self.width, self.height))
        # return img, boxes.astype(np.int32)

    def __len__(self):
        return len(self.imgIds)

    def trip(self, image, boxes, tbins):
        height, width = image.shape[:2]
        voyage = imutil.PlanarVoyage(height, width)

        imseq = []
        boxseq = []
        for _ in range(tbins):
            G_0to2 = voyage()
            out = self.img_warp(image, G_0to2, flags=cv2.INTER_LINEAR).astype(np.float32)
            tboxes = imutil.cv2_apply_transform_boxes(boxes, G_0to2)
            tboxes = clamp_boxes(tboxes, height, width)
            tboxes = discard_too_small(tboxes, 10)
            imseq.append(out)
            boxseq.append(tboxes)
        return imseq, boxseq

if __name__ == '__main__':
    dataDir = '/home/etienneperot/workspace/data/coco'
    dataType = 'val2017'
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

    mcoco = MovingCOCODataset(annFile)


    while 1:
        seq_im, seq_boxes = mcoco[np.random.randint(0, len(mcoco))]

        for im, boxes in zip(seq_im, seq_boxes):
            bboxes = vis.boxarray_to_boxes(boxes, boxes[:, -1], mcoco.catNms)
            img_ann = vis.draw_bboxes(im, bboxes, 2, colormap=cv2.COLORMAP_JET)

            cv2.imshow('im', im)
            cv2.waitKey()