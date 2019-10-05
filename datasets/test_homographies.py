from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2
from pycocotools.coco import COCO
from core.utils import vis
from core.utils import image as imutil
from functools import partial


def mask_instance_to_boxes(gt, anns, class_lookup):
    boxes = []
    labels = []
    for i in range(len(anns)):
        y, x = np.where(gt == (i + 1))
        if len(x) == 0 or len(y) == 0:
            continue
        x1, x2, y1, y2 = np.min(x), np.max(x), np.min(y), np.max(y)
        labels.append(class_lookup[anns[i]['category_id']])
        boxes.append(np.array([[x1, y1, x2, y2]]))
    if len(boxes) == 0:
        return np.array([[]]), []
    boxes = np.concatenate(boxes, axis=0).astype(np.int32)
    return boxes, labels


def wrap_boxes(boxes, labels, height, width):
    allbox = [boxes]
    alllbl = [labels]
    for y in [-height, 0, height]:
        for x in [-width, 0, width]:
            if x == 0 and y == 0:
                continue
            shift = boxes.copy()
            shift[:, [0, 2]] += x
            shift[:, [1, 3]] += y
            allbox.append(shift)
            alllbl.append(labels)
    return np.concatenate(allbox, 0), np.concatenate(alllbl, 0)


def boxarray_to_boxes(boxes):
    bboxes = []
    for i, box in enumerate(boxes):
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        bb = ('', i, pt1, pt2, None, None, None)
        bboxes.append(bb)
    return bboxes


def clamp_boxes(boxes, height, width):
    boxes[:, [0, 2]] = np.maximum(np.minimum(boxes[:, [0, 2]], width), 0)
    boxes[:, [1, 3]] = np.maximum(np.minimum(boxes[:, [1, 3]], height), 0)
    return boxes

def discard_too_small(boxes, labels, min_size=30):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    ids = np.where((w > min_size)*(h > min_size))
    return boxes[ids], labels[ids]


def show_voyage(img, anns, labelmap):
    height, width = img.shape[:2]

    voyage = imutil.PlanarVoyage(height, width)

    prev_img = img.astype(np.float32)


    mask = coco.annToMask(anns[0])
    for i in range(len(anns)):
        mask = np.maximum(mask, coco.annToMask(anns[i]))
    mask_rgb = cv2.applyColorMap((mask * 30) % 255, cv2.COLORMAP_HSV) * (mask > 0)[:, :, None].repeat(3, 2)
    mask_rgb = mask_rgb.astype(np.float32)/255.0


    # mask_inst = np.zeros((height, width), dtype=np.uint8)
    # for i in range(len(anns)):
    #     im = coco.annToMask(anns[i])
    #     mask_inst[im>0] = i+1

    class_lookup = np.array([0, 0, 0, 1])
    boxes = []
    labels = []
    for ann in anns:
        x, y, w, h = ann['bbox']
        label = ann['category_id']
        labels.append(label)
        boxes.append(np.array([[x, y, x + w, y + h]]))
    boxes = np.concatenate(boxes, axis=0)
    labels = class_lookup[np.array(labels)]

    boxes, labels = wrap_boxes(boxes, labels, height, width)


    img_warp = partial(cv2.warpPerspective, dsize=(width, height), borderMode=cv2.BORDER_WRAP)
    t = 0

    while 1:

        G_0to2 = voyage()

        out = img_warp(img, G_0to2, flags=cv2.INTER_LINEAR).astype(np.float32)

        out = (out - out.min()) / (out.max() - out.min())


        out_mask = img_warp(mask_rgb, G_0to2, flags=cv2.INTER_LINEAR).astype(np.float32)

        diff = out - prev_img

        # Remove some events
        # flow = get_flow(G_0to2, height, width)
        # flow = (flow-flow.min())/(flow.max()-flow.min())
        # viz_flow = flow_viz(flow)
        # cv2.imshow('flow', viz_flow)
        # im = (prev_img*255).astype(np.uint8)
        # ts = compute_timesurface(im, flow, diff).clip(0)
        # ts = filter_outliers(ts)
        # ts = (ts-ts.min())/(ts.max()-ts.min())
        # cv2.imshow('ts', ts)


        # warp & show boxes
        tboxes = imutil.cv2_apply_transform_boxes(boxes, G_0to2)
        #tboxes = imutil.cv2_clamp_filter_boxes(tboxes, G_0to2, (height, width))
        tboxes = clamp_boxes(tboxes, height, width)
        tboxes, tlabels = discard_too_small(tboxes, labels, 10)


        bboxes = vis.boxarray_to_boxes(tboxes.astype(np.int32), tlabels, labelmap)
        img_ann = vis.draw_bboxes(out.copy(), bboxes, 2, colormap=cv2.COLORMAP_JET)


        # Salt-and-Pepper
        # diff += (np.random.rand(height, width)[:,:,None].repeat(3,2) < 0.00001)/2
        # diff -= (np.random.rand(height, width)[:,:,None].repeat(3,2) < 0.00001) / 2
        #
        # diff *= np.random.rand(height, width, 3) < 0.9
        # diff = viz_diff(diff) + out_mask / 3

        img_ann = img_ann + out_mask/3

        cv2.imshow("diff", img_ann)
        #cv2.imshow("out", out)
        key = cv2.waitKey(5)
        if key == 27:
            break
        prev_img = out

        t += 1


if __name__ == '__main__':
    import glob
    #imgs = glob.glob("/home/etienneperot/workspace/data/coco/images/train2017/"+"*.jpg")

    dataDir = '/home/etienneperot/workspace/data/coco'
    dataType = 'val2017'
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    coco = COCO(annFile)

    catNms = ['person', 'car']
    catIds = coco.getCatIds(catNms=catNms)
    imgIds = coco.getImgIds(catIds=catIds)

    while 1:
        img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
        file_name = os.path.join(dataDir, 'images', dataType, img['file_name'])
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        image = cv2.imread(file_name)
        mask = coco.annToMask(anns[0])
        for i in range(len(anns)):
            mask += coco.annToMask(anns[i])


        show_voyage(image, anns, catNms)
