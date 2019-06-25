from __future__ import print_function
import time
import torch
import numpy as np
import cv2
cv2.setNumThreads(0)

def clamp_xyxy(x1, y1, x2, y2, width, height):
    x1 = np.minimum(np.maximum(x1, 0), width)
    x2 = np.minimum(np.maximum(x2, 0), width)
    y1 = np.minimum(np.maximum(y1, 0), height)
    y2 = np.minimum(np.maximum(y2, 0), height)
    return x1, y1, x2, y2


OK = 0
TOOSMALL = 1
TOOBIG = 2
TOP = 3
BOTTOM = 4
LEFT = 5
RIGHT = 6


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

    if x1 <= 0:
        flags[LEFT] = 1

    if x2 >= width - 1:
        flags[RIGHT] = 1

    if y1 <= 0:
        flags[TOP] = 1

    if y2 >= height:
        flags[BOTTOM] = 1

    if (x2 - x1) <= min_width or (y2 - y1) <= min_height:
        flags[TOOSMALL] = 1

    if (x2 - x1) >= width - 1 or (y2 - y1) >= height - 1:
        flags[TOOBIG] = 1

    return box, flags


class MovingSquare:
    """
     Responsible for endless MovingSquare
    """

    def __init__(self, t=10, h=300, w=300, c=1, max_stop=15):
        self.time, self.height, self.width = t, h, w
        self.minheight, self.minwidth = 30, 30
        self.stop_num = 0
        self.max_stop = max_stop
        self.class_id = np.random.randint(3)
        self.color = (0.5 + np.random.rand(c)/2).tolist()
        self.iter = 0
        self.reset()
        self.run()

    def reset(self):
        s = np.random.randint(self.minheight, 3 * self.minheight)
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
        # if np.random.rand() < 0.001 and v < 5 and self.iter > 10:
        #     self.stop_num = np.random.randint(1, self.max_stop)
        #
        if np.random.rand() < 0.01:
            self.reset_speed()

        x1, y1, x2, y2 = clamp_xyxy(self.x1, self.y1, self.x2, self.y2, self.width - 1, self.height - 1)

        self.iter += 1

        return (x1, y1, x2, y2)


class Animation:
    """
     Responsible for endless Animation
    """

    def __init__(self, t=10, h=300, w=300, c=1, max_stop=15, mode='none'):
        self.height = h
        self.width = w
        self.channels = c
        self.mode = mode
        self.t = t
        self.max_stop = max_stop

        self.num_objects = 1 #np.random.randint(0, 2)
        self.objects = []
        for i in range(self.num_objects):
            self.objects += [MovingSquare(t, h, w, c, max_stop)]
        self.reset()
        self.run()

    def reset(self):
        self.objects = []
        for i in range(self.num_objects):
            self.objects += [MovingSquare(self.t, self.height, self.width, self.channels, self.max_stop)]
        self.prev_img = np.zeros((self.height, self.width, self.channels), dtype=np.float32)
        self.img = np.zeros((self.height, self.width, self.channels), dtype=np.float32)

    def run(self):
        self.prev_img[...] = self.img

        self.img[...] = 0
        boxes = np.zeros((len(self.objects), 5), dtype=np.float32)
        for i, object in enumerate(self.objects):
            x1, y1, x2, y2 = object.run()

            if object.class_id == 0:
                cv2.rectangle(self.img, (x1, y1), (x2, y2), object.color, -1)
            elif object.class_id == 1:
                pt1 = x1, y2
                pt2 = x2, y2
                pt3 = (x1+x2)/2, y1
                triangle_cnt = np.array([pt1, pt2, pt3])
                cv2.drawContours(self.img, [triangle_cnt], 0, object.color, -1)
            else:
                ptc = (x1+x2)/2, (y1+y2)/2
                radius = int((x2-x1)/2)
                cv2.circle(self.img, ptc, radius, object.color, -1)
                
            boxes[i] = np.array([x1, y1, x2, y2, object.class_id])


        self.first_iteration = False

    
        output = np.transpose(self.img, [2, 0, 1])
        return output, boxes


class SquaresVideos:
    """
    Toy Detection DataBase for video detection.
    Move a Patch
    """

    def __init__(self, batchsize=32, t=10, h=300, w=300, c=3,
                 normalize=False, max_stops=30, mode='diff'):
        self.batchsize = batchsize
        self.num_frames = 100000
        self.channels = c
        self.time, self.height, self.width = t, h, w
        self.rate = 0
        self.normalize = normalize
        self.labelmap = ['square', 'triangle', 'circle']
        self.multi_aspect_ratios = False
        self.max_stops = max_stops
        self.animations = [Animation(t, h, w, c, self.max_stops, mode) for i in range(self.batchsize)]

    def reset(self):
        for anim in self.animations:
            anim.reset()

    def reset_size(self, height, width):
        self.height, self.width = height, width
        self.animations = [Animation(self.time, height, width, self.channels, self.max_stops) for i in
                           range(self.batchsize)]

    def __len__(self):
        return self.num_frames


    def next(self):
        x = torch.zeros(self.time, self.batchsize, self.channels, self.height, self.width)
        y = [[] for t in range(self.time)]

        for i, anim in enumerate(self.animations):
            for t in range(self.time):
                im, boxes = anim.run()
                x[t, i, :] = torch.from_numpy(im)
                y[t].append(torch.from_numpy(boxes))

        return x, y

    def __iter__(self):
        for _ in range(len(self)):
            yield self.next()


if __name__ == '__main__':
    from core.utils import boxarray_to_boxes, draw_bboxes, make_single_channel_display

    dataloader = SquaresVideos(t=10, c=3, h=128, w=128, batchsize=1)

    start = 0

    for i, (x, y) in enumerate(dataloader):
        start = time.time()
        for j in range(x.size(1)):

            for t in range(x.size(0)):
                boxes = y[t][j].cpu()
                bboxes = boxarray_to_boxes(boxes[:, :4], boxes[:, 4], dataloader.labelmap)


                img = x[t, j, :].numpy().astype(np.float32)
                if img.shape[0] == 1:
                    img = make_single_channel_display(img[0], -1, 1)
                else:
                    img = np.moveaxis(img, 0, 2)
                    show = np.zeros((dataloader.height, dataloader.width, 3), dtype=np.float32)
                    show[...] = img
                    img = show

                img = draw_bboxes(img, bboxes)
                cv2.imshow('example', img)
                key = cv2.waitKey(5)
                if key == 27:
                    exit()
