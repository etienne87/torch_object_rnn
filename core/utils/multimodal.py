from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import cv2


def i_xor_b_and_o(i, b, o):
    return np.logical_xor(i, np.logical_and(b, o))


def multimodal_annotate(boxes, height=256, width=256, border_thickness=1, shift=2):
    inout = np.zeros((height>>shift, width>>shift), dtype=np.uint8)
    boundaries = np.zeros((height>>shift, width>>shift), dtype=np.uint8)
    semantics = np.zeros((height>>shift, width>>shift), dtype=np.uint8)

    #sort boxes (always annotate smaller boxes first)
    wh = boxes[:, :2]-boxes[:, 2:4]
    wh = wh[:,0]*wh[:,1]
    idx = np.argsort(wh)[::-1]
    boxes = boxes[idx]

    scale = 2**shift
    for box in boxes:
        x1, y1, x2, y2, label = box.tolist()
        if shift > 0:
            x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
        ptc = int((x1 + x2) / 2), int((y1 + y2) / 2)
        shape = int((x2-x1)/2), int((y2-y1)/2)
        tmp = np.zeros((height>>shift, width>>shift), dtype=np.uint8)
        cv2.ellipse(tmp, ptc, shape, 0, 0, 360, 1, -1)
        inout += tmp    
        cv2.ellipse(semantics, ptc, shape, 0, 0, 360, int(label), -1)
        cv2.ellipse(boundaries, ptc, shape, 0, 0, 360, 1, border_thickness)

    inside = (inout == 1)
    outside = (inout > 1) 
    boundaries = boundaries > 0
    outside = np.logical_and(outside, boundaries)
    ans = [inside, boundaries, outside, semantics]
    show_multimodal(*ans)
    return ans

def get_bounding_boxes(i, b, o, s):
    ibo = i_xor_b_and_o(i, b, o)
    contours, hierarchy = cv2.findContours(ibo.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for i in range(len(contours)):
        xy = contours[i][:,0,:]
        x1, y1 = xy[:,0].min(), xy[:, 1].min()
        x2, y2 = xy[:,0].max(), xy[:, 1].max()
        boxes.append(np.array([x1,y1,x2,y2]))
    return boxes, contours

def show_multimodal(inside, boundaries, outside, semantics):
    ibo = i_xor_b_and_o(inside, boundaries, outside)
    contours, hierarchy = cv2.findContours(ibo.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ibo = ibo[..., None].repeat(3, 2).astype(np.uint8) * 255
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    colors = [tuple(*item) for item in colors.tolist()]
    random.shuffle(colors)
    for i in range(len(contours)):
        xy = contours[i][:,0,:]
        x1, y1 = xy[:,0].min(), xy[:, 1].min()
        x2, y2 = xy[:,0].max(), xy[:, 1].max()
        cv2.rectangle(ibo, (x1,y1), (x2,y2), colors[i], 2)
        ibo = cv2.drawContours(ibo, [contours[i]], -1, colors[i*4], 3)

    #semantic
    sem_img = cv2.applyColorMap((semantics*30).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    sem_img[semantics==0] = 0

    #draw on img
    # img[..., 0] = np.minimum(1.0, (img[..., 0] + inside*0.5))
    # img[..., 2] = np.minimum(1.0, (img[..., 2] + boundaries*0.5))
    # cv2.imshow('img', img[..., ::-1])
    
    cv2.imshow('ibo', ibo)
    cv2.imshow('sem_img', sem_img)
    cv2.waitKey()

if __name__ == '__main__':
    height, width = 256, 256
  
    box_width, box_height = 64, 64
    
    # Case 1 (one box inside another)
    img = np.zeros((height, width, 3), dtype=np.float32)
    boxes = np.array([[128-box_width, 128-box_height, 128+box_width, 128+box_height, 5],
                      [128-box_width/2, 128-box_height/2, 128+box_width/2, 128+box_height/2, 2]])
    multimodal_annotate(img, boxes)


    # Case 2 (one box overlaps another)
    img = np.zeros((height, width, 3), dtype=np.float32)
    boxes = np.array([[128-box_width, 128-box_height, 128+box_width, 128+box_height, 7],
                      [128, 128-box_height, 128+box_width, 128, 4]])
    multimodal_annotate(img, boxes)

    # Case 3 (no box overlaps)
    img = np.zeros((height, width, 3), dtype=np.float32)
    boxes = np.array([[128-box_width*2, 128-box_height, 128, 128+box_height, 8],
                      [128, 128-box_height, 128+box_width, 128, 9]])
    multimodal_annotate(img, boxes)

    # Case 4 (complicated flower overlaps)
    img = np.zeros((height, width, 3), dtype=np.float32)
    central_box = np.array([128-box_width, 128-box_height, 128+box_width, 128+box_height, 1])

    box_width2, box_height2 = 64, 64
    radius = box_width #np.sqrt(box_width**2 + box_height**2) #could also just be box_width
    num_petals = 6
    boxes = [central_box[None,:]]
    for i in range(num_petals):
        alpha = (2 * np.pi) / num_petals * i
        x = radius * np.cos(alpha) 
        y = radius * np.sin(alpha)
        cx, cy = 128 + x, 128 + y
        box = np.array([cx-box_width2/2, cy-box_height2/2, cx+box_width2/2, cy+box_height2/2, i+2])
        boxes.append(box[None,:])
    boxes = np.concatenate(boxes)
    multimodal_annotate(img, boxes)
