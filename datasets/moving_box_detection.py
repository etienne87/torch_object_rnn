from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
from torch.utils.data import Dataset
from core.utils import opts

import numpy as np
import cv2
import math
cv2.setNumThreads(0)



OK = 0
TOOSMALL = 1
TOOBIG = 2
TOP = 3
BOTTOM = 4
LEFT = 5
RIGHT = 6
STOP_AT_BEGINNING = False


def clamp_xyxy(x1, y1, x2, y2, width, height):
    x1 = np.minimum(np.maximum(x1, 0), width)
    x2 = np.minimum(np.maximum(x2, 0), width)
    y1 = np.minimum(np.maximum(y1, 0), height)
    y2 = np.minimum(np.maximum(y2, 0), height)
    return x1, y1, x2, y2


def rotate(x,y,xo,yo,theta):
    xr=math.cos(theta)*(x-xo)-math.sin(theta)*(y-yo) + xo
    yr=math.sin(theta)*(x-xo)+math.cos(theta)*(y-yo) + yo
    return [xr,yr]


def move_box(x1, y1, x2, y2, vx, vy, vs, width, height, min_width, min_height):
    x1, x2, y1, y2 = x1 + vx, x2 + vx, y1 + vy, y2 + vy

    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1), (y2 - y1)
    w *= vs
    h *= vs
    x1, x2, y1, y2 = int(np.round(xc - w / 2)), int(np.round(xc + w / 2)), \
                     int(np.round(yc - h / 2)), int(np.round(yc + h / 2))

    box = (x1, y1, x2, y2)
    flags = {}

    if x1 < 0:
        flags[LEFT] = 1

    if x2 > width - 1:
        flags[RIGHT] = 1

    if y1 < 0:
        flags[TOP] = 1

    if y2 > height:
        flags[BOTTOM] = 1

    if (x2 - x1) <= min_width or (y2 - y1) <= min_height:
        flags[TOOSMALL] = 1

    if (x2 - x1) >= width or (y2 - y1) >= height:
        flags[TOOBIG] = 1

    return box, flags


class MovingSquare(object):
    """
     Responsible for endless MovingSquare
    """

    def __init__(self, t=10, h=300, w=300, c=1, max_stop=15, max_classes=3):
        self.time, self.height, self.width = t, h, w
        self.aspect_ratio = 1.0 #min(3, max(1./3, np.random.randn()/3 + 1.0))
        self.minheight, self.minwidth = 30, 30
        self.stop_num = 0
        self.max_stop = max_stop
        self.class_id = np.random.randint(max_classes)
        self.color = (0.5 + np.random.rand(c)/2).tolist()
        self.iter = 0
        self.reset()
        self.run()

        if STOP_AT_BEGINNING:
            self.stop_num = np.random.randint(10, 30)

    def reset(self):
        s = np.random.randint(self.minheight, self.height-self.minheight)
        xmax = np.maximum(self.width - 1 - s, 1)
        ymax = np.maximum(self.height - 1 - s, 1)
        self.x1 = np.random.randint(0, xmax)
        self.y1 = np.random.randint(0, ymax)
        self.x2 = self.x1 + s
        self.y2 = self.y1 + s

        self.reset_speed()
        self.first_iteration = True

    def reset_speed(self):
        max_v = 10
        self.vx = np.random.randint(1, max_v) * (np.random.randint(0, 2) * 2 - 1)
        self.vy = np.random.randint(1, max_v) * (np.random.randint(0, 2) * 2 - 1)
        self.vs = np.random.uniform(0.02, 0.05)
        self.va = np.random.randint(5, 20)
        if np.random.randint(0, 2) == 0:
            self.vs = 1 + self.vs
        else:
            self.vs = 1 - self.vs

    def run(self):
        return self.run_random()

    def run_circular(self):
        h, w = self.y2-self.y1, self.x2-self.x1
        yc, xc = (self.y2+self.y1)/2, (self.x2+self.x1)/2
        xc, yc = rotate(xc, yc, self.width/2, self.height/2, math.pi / self.va)
        self.x1, self.y1 = int(xc-w/2), int(yc-h/2)
        self.x2, self.y2 = int(xc+w/2), int(yc+h/2)
        self.iter += 1
        return self.x1, self.y1, self.x2, self.y2

    def run_random(self):
        if self.stop_num == 0:

            box, flags = move_box(self.x1, self.y1, self.x2, self.y2,
                                  self.vx, self.vy, self.vs,
                                  self.width, self.height, self.minwidth, self.minheight)


            if TOOSMALL in flags:
                self.vs = 1 + np.random.uniform(0.02, 0.05)

            if TOOBIG in flags:
                self.vs = 1 - np.random.uniform(0.02, 0.05)

            if LEFT in flags:
                self.vx = np.random.randint(1, 7)

            if RIGHT in flags:
                self.vx = -np.random.randint(1, 7)

            if BOTTOM in flags:
                self.vy = -np.random.randint(1, 7)

            if TOP in flags:
                self.vy = np.random.randint(1, 7)


            self.x1, self.y1, self.x2, self.y2 = box
        else:
            self.stop_num -= 1

        v = np.sqrt(self.vx ** 2 + self.vy ** 2 + self.vs ** 2)
        # if np.random.rand() < 0.001 and v < 5 and self.iter > 10:
        #     self.stop_num = np.random.randint(1, self.max_stop)
        # if np.random.rand() < 0.01:
        #     self.reset_speed()

        xc = (self.x1 + self.x2)/2
        yc = (self.y1 + self.y2)/2
        w = (self.x2-self.x1)
        h = w * self.aspect_ratio
        self.x1 = int(xc - w/2)
        self.x2 = int(xc + w/2)
        self.y1 = int(yc - h/2)
        self.y2 = int(yc + h/2)

        x1, y1, x2, y2 = clamp_xyxy(self.x1, self.y1, self.x2, self.y2, self.width, self.height)
        #x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2

        self.iter += 1


        return (x1, y1, x2, y2)


class Animation(object):
    """
     Responsible for endless Animation
    """

    def __init__(self, t=10, h=300, w=300, c=1, max_stop=15, mode='none', max_classes=1, max_objects=3, render=True):
        #super(object, self).__init__()
        self.height = h
        self.width = w
        self.channels = c
        self.mode = mode
        self.t = t
        self.max_stop = max_stop
        self.max_classes = max_classes
        self.max_objects = max_objects
        self.num_objects = np.random.randint(1, max_objects + 1)
        self.objects = []
        self.render = render
        for i in range(self.num_objects):
            self.objects += [MovingSquare(t, h, w, c, max_stop, max_classes=max_classes)]
        self.reset()
        self.run()
        self.prev_boxes = None
        self.first_iteration = True

    def reset(self):
        self.objects = []
        self.num_objects = np.random.randint(1, self.max_objects + 1)
        for i in range(self.num_objects):
            self.objects += [MovingSquare(self.t, self.height, self.width, self.channels, self.max_stop,
                                          max_classes=self.max_classes)]
        self.prev_img = np.zeros((self.height, self.width, self.channels), dtype=np.float32)
        self.img = np.zeros((self.height, self.width, self.channels), dtype=np.float32)
        self.first_iteration = True

    def run(self):
        if self.render:
            self.prev_img[...] = self.img
            self.img[...] = 0

        boxes = np.zeros((len(self.objects), 5), dtype=np.float32)
        for i, object in enumerate(self.objects):
            x1, y1, x2, y2 = object.run()
            boxes[i] = np.array([x1, y1, x2, y2, object.class_id])

            if not self.render:
                continue

            if object.class_id == 0:
                cv2.rectangle(self.img, (x1, y1), (x2, y2), object.color, -1)
            elif object.class_id == 1:
                pt1 = x1, y2
                pt2 = x2, y2
                pt3 = (x1+x2)/2, y1
                triangle_cnt = np.array([pt1, pt2, pt3])
                cv2.drawContours(self.img, [triangle_cnt], 0, object.color, -1)
            else:
                ptc = (x1 + x2) / 2, (y1 + y2) / 2
                shape = (x2-x1)/2, (y2-y1)/2
                cv2.ellipse(self.img, ptc, shape, 0, 0, 360, object.color, -1)

        if not self.render:
            return None, boxes


        if self.mode == 'diff':
            output = self.img - self.prev_img
            output = np.transpose(output, [2, 0, 1])
            if self.first_iteration:
                output[...] = 0
        else:
            output = np.transpose(self.img, [2, 0, 1])

        self.first_iteration = False
        return output, boxes


class SquaresVideos(Dataset):
    """
    Toy Detection DataBase for video detection.
    Move a Patch
    """

    def __init__(self, batchsize=32, t=10, h=300, w=300, c=3,
                 normalize=False, max_stops=30, max_objects=1, max_classes=1, mode='diff', render=True):
        self.batchsize = batchsize
        self.num_batches = 500
        self.channels = c
        self.time, self.height, self.width = t, h, w
        self.rate = 0
        self.normalize = normalize
        self.labelmap = ['square', 'triangle', 'circle']
        self.multi_aspect_ratios = False
        self.max_stops = max_stops
        self.mode = mode
        self.render = render
        self.max_objects = max_objects
        self.max_classes = max_classes
        self.max_consecutive_batches = 2**10
        self.build()

    def build(self):
        num_sets = max(1, self.num_batches // self.max_consecutive_batches)
        self.animations = [[Animation(self.time, self.height, self.width, self.channels, self.max_stops, self.mode,
                                     max_objects=self.max_objects,
                                     max_classes=self.max_classes,
                                     render=self.render) for _ in range(self.batchsize)]
                                                         for _ in range(num_sets)]

    #TO BE CALLED ONLY BETWEEN EPOCHS
    def reset(self):
        for j in range(len(self.animations)):
            for i in range(len(self.animations[j])):
                self.animations[j][i].reset()

    # TO BE CALLED ONLY BETWEEN EPOCHS
    def reset_size(self, height, width):
        self.height, self.width = height, width
        self.build()

    def __len__(self):
        return self.num_batches * self.batchsize

    def __getitem__(self, item):
        num_batch = (item // self.batchsize)
        num_set = min(num_batch // self.max_consecutive_batches, len(self.animations)-1)
        reset = num_batch%self.max_consecutive_batches == 0
        anim = self.animations[num_set][item%self.batchsize]
        x = torch.zeros(self.time, self.channels, self.height, self.width)
        y = []
        for t in range(self.time):
            im, boxes = anim.run()
            if self.render:
                x[t] = torch.from_numpy(im)
            y.append(torch.from_numpy(boxes))

        return x, y, reset

    # def next(self):
    #     x = torch.zeros(self.time, self.batchsize, self.channels, self.height, self.width)
    #     y = [[] for t in range(self.time)]
    #     r = []
    #
    #     if not self.dataset.render:
    #         x = None
    #
    #     for i, anim in enumerate(self.dataset.animations):
    #         for t in range(self.time):
    #             im, boxes = anim.run()
    #             if self.render:
    #                 x[t, i, :] = torch.from_numpy(im)
    #             y[t].append(torch.from_numpy(boxes))
    #
    #     return x, y


# class MovingAnimationLoader(object):
#     def __init__(self, dataset):
#         self.dataset = dataset
#
#     def __iter__(self):
#         for _ in range(len(self)):
#             yield self.next()


if __name__ == '__main__':
    from core.utils.vis import boxarray_to_boxes, draw_bboxes, make_single_channel_display

    dataset = SquaresVideos(t=10, c=1, h=256, w=256, batchsize=2, mode='diff', render=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0,
                           shuffle=False, collate_fn=opts.video_collate_fn_with_reset_info, pin_memory=True)


    start = 0

    for x, y, r in dataloader:
        if r:
            print('Reset !')
        for t in range(len(y)):
            for j in range(dataset.batchsize):
                boxes = y[t][j].cpu()

                boxes = boxes.cpu().numpy().astype(np.int32)
                bboxes = boxarray_to_boxes(boxes[:, :4], boxes[:, 4], dataset.labelmap)

                if dataset.render:
                    img = x[t, j, :].numpy().astype(np.float32)
                    if img.shape[0] == 1:
                        img = make_single_channel_display(img[0], -1, 1)
                    else:
                        img = np.moveaxis(img, 0, 2)
                        show = np.zeros((dataset.height, dataset.width, 3), dtype=np.float32)
                        show[...] = img
                        img = show
                else:
                    img = np.zeros((256,256,3), dtype=np.uint8)

                img = draw_bboxes(img, bboxes)

                cv2.imshow('example'+str(j), img)
                key = cv2.waitKey(5)
                if key == 27:
                    exit()
