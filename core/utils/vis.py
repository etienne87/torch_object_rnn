from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def boxarray_to_boxes(boxes, labels, labelmap):
    bboxes = []
    for label, box in zip(labels, boxes):
        class_name = labelmap[label]
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        bb = (class_name, label, pt1, pt2, None, None, None)
        bboxes.append(bb)
    return bboxes


def general_frame_display(im):
    if im.shape[0] == 1:
        return make_single_channel_display(im[0], -1, 1)
    else:
        im = np.transpose(im, [1, 2, 0]).copy()
        im = ((im - im.min()) / (im.max() - im.min()) * 255).astype(np.uint8)
        return im


def make_single_channel_display(img, low=None, high=None):
    if low is None:
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    else:
        img = ((img+1)/2*255).astype(np.uint8)
    img = (img.astype(np.uint8))[..., None].repeat(3, 2)
    return img


def single_frame_display(im):
    return make_single_channel_display(im[0], -1, 1)


def draw_bboxes(img, bboxes, thickness=1, colormap=cv2.COLORMAP_HSV, colordefault=None):
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), colormap)

    colors = [tuple(*item) for item in colors.tolist()]
    for bbox in bboxes:
        class_name, class_id, pt1, pt2, _, _, _ = bbox
        center = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
        color = colors[(class_id * 40)%255] if colordefault is None else colordefault
        cv2.rectangle(img, pt1, pt2, color, thickness)
        cv2.putText(img, class_name, (center[0], max(0, pt2[1] - 1) ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
    return img


def filter_outliers(g, num_std=2):
    grange = num_std*g.std()
    gimg_min = g.mean() - grange
    gimg_max = g.mean() + grange
    g_normed = np.minimum(np.maximum(g,gimg_min),gimg_max) #clamp
    return g_normed


def write_video_opencv(video_name, video_tensor):
    frames, height, width, c = video_tensor.shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))
    for i in range(frames):
        out_video.write(video_tensor[i])
    out_video.release()


def draw_txn_boxes_on_images(images, targets, grid, make_img_fun, period, time, ncols, labelmap):
    for t in range(len(targets)):
        for i in range(len(targets[t])):
            y = i // ncols
            x = i % ncols
            img = make_img_fun(images[t, i])
            boxes, labels, scores = targets[t][i]
            if boxes is not None:
                boxes = boxes.cpu().numpy().astype(np.int32)
                labels = labels.cpu().numpy().astype(np.int32)
                bboxes = boxarray_to_boxes(boxes, labels, labelmap)
                img = draw_bboxes(img, bboxes)
            grid[t + period * time, y, x] = img