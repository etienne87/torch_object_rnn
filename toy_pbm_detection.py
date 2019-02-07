from __future__ import print_function
import sys
import torch
import numpy as np
import cv2


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


class SquaresImages:
    """
    Toy Detection DataBase for image detection.
    Make a Rectangle filled with white or black and its bbox
    """

    def __init__(self, batchsize=32, h=300, w=300, normalize=False, cuda=False):
        self.batchsize = batchsize
        self.num_frames = 100000
        self.height, self.width = h, w
        self.minheight, self.minwidth = 40, 40
        self.cuda = cuda
        self.rate = 0
        self.normalize = normalize
        self.labelmap = ['square']
        self.multi_aspect_ratios = False

    def cuda(self, cuda):
        self.cuda = cuda

    def __len__(self):
        return self.num_frames

    def next(self):
        x = torch.zeros(self.batchsize, 1, self.height, self.width)
        y = []
        for i in range(self.batchsize):

            if self.multi_aspect_ratios:
                x1 = np.random.randint(0, self.width - 1 - self.minwidth)
                y1 = np.random.randint(0, self.height - 1 - self.minheight)
                x2 = np.random.randint(x1 + self.minwidth, self.width)
                y2 = np.random.randint(y1 + self.minheight, self.height)
            else:
                s = np.random.randint(self.minheight, 4 * self.minheight)
                xmax = np.maximum(self.width - 1 - s, 1)
                ymax = np.maximum(self.height - 1 - s, 1)
                x1 = np.random.randint(0, xmax)
                y1 = np.random.randint(0, ymax)
                x2 = x1 + s
                y2 = y1 + s

            x1_, y1_, x2_, y2_ = clamp_xyxy(x1, y1, x2, y2, self.width - 1, self.height - 1)

            x[i, 0, y1_:y2_, x1_:x2_] = 1
            # assert (x[i].min() != x[i].max())

            # Normalize (0,1)
            if self.normalize:
                x1, x2 = float(x1) / self.width, float(x2) / self.width
                y1, y2 = float(y1) / self.height, float(y2) / self.height
            box = np.array([x1, y1, x2, y2, 0], dtype=np.float32).reshape(1, 5)
            y.append(torch.from_numpy(box))

        return x, y


class MovingSquare:
    """
     Responsible for endless MovingSquare
    """

    def __init__(self, t=10, h=300, w=300, c=1, max_stop=15):
        self.time, self.height, self.width = t, h, w
        self.minheight, self.minwidth = 30, 30
        self.stop_num = 0
        self.max_stop = max_stop
        self.reset()
        self.run()
        self.first_iteration = True
        self.color = 0.5 + torch.rand(c) / 2

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
        if np.random.rand() < 0.01 and v < 5 and not self.first_iteration:
            self.stop_num = np.random.randint(5, self.max_stop)

        if np.random.rand() < 0.01:
            self.reset_speed()

        x1, y1, x2, y2 = clamp_xyxy(self.x1, self.y1, self.x2, self.y2, self.width - 1, self.height - 1)

        self.first_iteration = False

        return (x1, y1, x2, y2)


class SquareAnimation:
    """
     Responsible for endless WhiteSquare Animation
    """

    def __init__(self, t=10, h=300, w=300, c=1, max_stop=15):
        self.height = h
        self.width = w
        self.channels = c

        self.num_objects = 1 #np.random.randint(1, 5)
        self.objects = []
        for i in range(self.num_objects):
            self.objects += [MovingSquare(t, h, w, c, max_stop)]
        self.reset()
        self.run()

    def reset(self):
        for object in self.objects:
            object.reset()
        self.prev_img = torch.zeros(self.channels, self.height, self.width)
        self.img = torch.zeros(self.channels, self.height, self.width)
        self.flow = torch.zeros(2, self.height, self.width)


    def run(self):
        self.prev_img[...] = self.img

        self.img[...] = 0
        boxes = np.zeros((len(self.objects), 5), dtype=np.float32)
        for i, object in enumerate(self.objects):
            x1, y1, x2, y2 = object.run()
            # assert(object.color.size(0) == self.channels)
            # print(object.color.size(0), ' channels: ', self.channels)
            for c in range(self.channels):
                self.img[c, y1:y2, x1:x2] += object.color[c]
            boxes[i] = np.array([x1, y1, x2, y2, 0])

        # self.img += (torch.rand(self.channels,self.height,self.width)>0.99).float()

        diff = self.img - self.prev_img

        self.first_iteration = False

        return self.img, boxes


class SquaresVideos:
    """
    Toy Detection DataBase for video detection.
    Make a Rectangle filled with white or black and its bbox
    The Rectangle moves continuously, so network can infer motion.
    This simulates DVS events Histograms
    """

    def __init__(self, batchsize=32, t=10, h=300, w=300, c=3, normalize=False, cuda=False, encode_all_timesteps=False):
        self.batchsize = batchsize
        self.num_frames = 100000
        self.channels = c
        self.time, self.height, self.width = t, h, w
        self.cuda = cuda
        self.rate = 0
        self.normalize = normalize
        self.labelmap = ['square']
        self.multi_aspect_ratios = False
        self.max_stops = 6
        self.animations = [SquareAnimation(t, h, w, c, self.max_stops) for i in range(self.batchsize)]
        self.encode_all_timesteps = encode_all_timesteps

    def reset(self):
        for anim in self.animations:
            anim.reset()

    def reset_size(self, height, width):
        self.height, self.width = height, width
        self.animations = [SquareAnimation(self.time, height, width, self.channels, self.max_stops) for i in
                           range(self.batchsize)]

    def cuda(self, cuda):
        self.cuda = cuda

    def __len__(self):
        return self.num_frames

    def __get_item__(self, item):


    def next(self):
        x = torch.zeros(self.time, self.batchsize, self.channels, self.height, self.width)
        if self.encode_all_timesteps:
            y = [[] for t in range(self.time)]
        else:
            y = []

        for i, anim in enumerate(self.animations):
            for t in range(self.time):
                im, boxes = anim.run()
                x[t, i, :] = im
                if self.encode_all_timesteps:
                    y[t].append(torch.from_numpy(boxes))

            if not self.encode_all_timesteps:
                y.append(torch.from_numpy(boxes))

        return x, y


if __name__ == '__main__':
    sys.path.insert(0, '../')
    from utils import boxarray_to_boxes, draw_bboxes, make_single_channel_display


    anim = SquareAnimation(c=1, max_stop=80)
    while(1):
        x, boxes = anim.run()

        if x.size(0) == 1:
            img = make_single_channel_display(x.numpy()[0], -1, 1)
        else:
            x = x.numpy()
            img = np.moveaxis(x,0,2)

        cv2.imshow('img',img)
        cv2.waitKey(10)

    tp = SquaresVideos(batchsize=1, h=240, w=304)

    print('channels: ', tp.channels)

    for i in range(100):

        if i % 2 == 0:
            scale = 2 ** np.random.randint(0, 3)
            print('new scale: ', scale)
            H, W = 240 / scale, 304 / scale
            tp.reset_size(H, W)

        x, y = tp.next()
        x = x.cpu()

        # Display It!
        for j in range(x.size(0)):
            boxes = y[j].cpu().numpy().astype(np.float32)
            boxes = boxes.astype(np.int32)
            bboxes = boxarray_to_boxes(boxes)

            print('boxes: ', boxes)

            if isinstance(tp, SquaresVideos):
                for t in range(x.size(2)):
                    img = x[j, :, t].numpy().astype(np.float32)
                    if img.shape[0] == 1:
                        img = make_single_channel_display(img[0], -1, 1)
                    else:
                        img = np.moveaxis(img, 0, 2)
                        show = np.zeros((H, W, 3), dtype=np.float32)
                        show[...] = img
                        img = show

                    img = draw_bboxes(img, bboxes)
                    cv2.imshow('example', img)
                    key = cv2.waitKey()
                    if key == 27:
                        exit()
            else:
                img = x[j, 0].numpy().astype(np.float32)
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                img = (img.astype(np.uint8))[..., None].repeat(3, 2)
                img = draw_bboxes(img, bboxes)

                cv2.imshow('example', img)
                key = cv2.waitKey()
                if key == 27:
                    exit()
