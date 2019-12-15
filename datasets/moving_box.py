from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import numpy as np
import random
import cv2
#  cv2.setNumThreads(0)

OK = 0
TOOSMALL = 1
TOOBIG = 2
TOP = 3
BOTTOM = 4
LEFT = 5
RIGHT = 6

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

    def __init__(self, h=300, w=300, max_stop=15, max_classes=3):
        self.height, self.width = h, w
        self.aspect_ratio = 1.0 
        self.minheight, self.minwidth = 30, 30
        self.stop_num = 0
        self.max_stop = max_stop
        self.class_id = np.random.randint(max_classes)
        self.iter = 0
        self.reset()
        self.run()

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
        if np.random.rand() < 0.001 and v < 5 and self.iter > 10:
            self.stop_num = np.random.randint(1, self.max_stop)
        if np.random.rand() < 0.01:
            self.reset_speed()

        xc = (self.x1 + self.x2)/2
        yc = (self.y1 + self.y2)/2
        w = (self.x2-self.x1)
        h = w * self.aspect_ratio
        self.x1 = int(xc - w/2)
        self.x2 = int(xc + w/2)
        self.y1 = int(yc - h/2)
        self.y2 = int(yc + h/2)

        x1, y1, x2, y2 = clamp_xyxy(self.x1, self.y1, self.x2, self.y2, self.width, self.height)
        self.iter += 1
        return (x1, y1, x2, y2)


class Animation(object):
    """
    Responsible for endless Animation
    """
    def __init__(self, height, width, channels, max_stop=15, max_classes=1, max_objects=3):
        self.height, self.width, self.channels = height, width, channels
        self.max_stop = max_stop
        self.max_classes = max_classes
        self.max_objects = max_objects
        self.num_objects = np.random.randint(1, max_objects + 1)
        self.objects = []
        for i in range(self.num_objects):
            self.objects += [MovingSquare(self.height, self.width, max_stop, max_classes=max_classes)]
        self.reset()
        self.run()
        self.prev_boxes = None
        self.first_iteration = True
        
    def reset(self):
        self.objects = []
        self.num_objects = np.random.randint(1, self.max_objects + 1)
        self.img = np.zeros((self.height, self.width, self.channels), dtype=np.float32)
        for i in range(self.num_objects):
            self.objects += [MovingSquare(self.height, self.width, self.max_stop,
                                          max_classes=self.max_classes)]
        self.first_iteration = True

    def run(self):
        raise NotImplementedError()