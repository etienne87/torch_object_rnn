from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from core.eval import recall
from core.eval import mean_ap


def random_box(minheight, minwidth, height, width):
    cx = np.random.randint(minwidth, width - minwidth)
    cy = np.random.randint(minheight, height - minheight)
    w = np.random.randint(minwidth/2, min(width - cx, cx))
    h = np.random.randint(minwidth/2, min(height - cy, cy))
    x1 = cx - w
    x2 = cx + w
    y1 = cy - h
    y2 = cy + h
    return x1, y1, x2, y2


def generate_dummy_imgs(num_imgs=10, height=512, width=512, max_box=10, num_classes=1):
    gts = []
    proposals = []
    minheight, minwidth = 30, 30
    for i in range(num_imgs):
        num_boxes = np.random.randint(0, max_box + 1)
        num_boxes_pred = num_boxes + np.random.randint(0, 2)
        gt = np.zeros((num_boxes, 5), dtype=np.float32)
        pred = np.zeros((num_boxes_pred, 6), dtype=np.float32)

        for j in range(num_boxes):
            gt[j, :4] = random_box(minheight, minwidth, height, width)
            gt[j, 4] = np.random.randint(0, num_classes)

        for k in range(num_boxes_pred):
            if k < num_boxes and np.random.randint(0, 10) < 9:
                w, h = (gt[k, 2] - gt[k, 0]), (gt[k, 3] - gt[k, 1])
                s = min(w, h) / 16.0
                cxy = gt[k, :2] + np.array([w/2, h/2]) + np.random.uniform(-s, s, 2)
                wh = np.array([w, h]) * np.random.uniform(1 - 1./32, 1 + 1./32)
                pt1 = cxy - wh/2
                pt2 = cxy + wh/2
                pred[k, :4] = np.array([pt1[0], pt1[1], pt2[0], pt2[1]])
                pred[k, 4] = np.random.uniform(0.1, 1.0)
                pred[k, 5] = gt[k, 4]
            else:
                pred[k, :4] = random_box(minheight, minwidth, height, width)
                pred[k, 4] = np.random.uniform(0.0, 0.2)
                pred[k, 5] = np.random.randint(0, num_classes)

        gts.append(gt)
        proposals.append(pred)

    return gts, proposals


def show_gt_pred(gts, preds, height=512, width=512):
    for gt, pred in zip(gts, preds):
        img = np.zeros((height, width, 3), dtype=np.uint8)

        for j in range(gt.shape[0]):
            cv2.rectangle(img, (gt[j, 0], gt[j, 2]), (gt[j, 1], gt[j, 3]), (255, 0, 0), 1)
        for k in range(pred.shape[0]):
            cv2.rectangle(img, (pred[k, 0], pred[k, 2]), (pred[k, 1], pred[k, 3]), (0, 0, 255), 1)

        cv2.imshow('img', img)
        key = cv2.waitKey()
        if key == 27:
            break

if __name__ == '__main__':

    num_classes = 2
    gts, proposals = generate_dummy_imgs(1000, num_classes=num_classes)
    # show_gt_pred(gts, proposals)
    #
    # recalls = recall.eval_recalls(gts,
    #              proposals,
    #              proposal_nums=3,
    #              iou_thrs=[0.3, 0.5],
    #              print_summary=True)

    det_results, gt_bboxes, gt_labels = mean_ap.convert(gts, proposals, num_classes)


    uns = np.unique(np.concatenate([item[:, None] for item in gt_labels]))
    print("unique classes: ", uns)

    mean_ap, eval_results = mean_ap.eval_map(det_results,
                                             gt_bboxes,
                                             gt_labels,
                                             gt_ignore=None,
                                             scale_ranges=None,
                                             iou_thr=0.9,
                                             dataset=None,
                                             print_summary=True)