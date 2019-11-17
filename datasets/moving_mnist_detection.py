from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from core.utils import opts
import datasets.moving_box_detection as toy


TRAIN_DATASET = datasets.MNIST('../data', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ]))

TEST_DATASET = datasets.MNIST('../data', train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ]))

class MovingMnistAnimation(toy.Animation):
    def __init__(self, t=10, h=128, w=128, c=1, max_stop=15,
                max_objects=3, train=True):
        self.dataset_ = TRAIN_DATASET if train else TEST_DATASET
        self.label_offset = 1
        super(MovingMnistAnimation, self).__init__(t, h, w, c, max_stop, 'none', 10, max_objects, True)

    def reset(self):
        super(MovingMnistAnimation, self).reset()
        for i in range(len(self.objects)):
            idx = np.random.randint(0, len(self.dataset_))
            x, y = self.dataset_[idx]
            self.objects[i].class_id = y
            self.objects[i].idx = idx
            img = x.numpy()[0]
            img = (img-img.min())/(img.max()-img.min())
            abs_img = np.abs(img)
            y, x = np.where(abs_img > 0.45)
            x1, x2 = np.min(x), np.max(x)
            y1, y2 = np.min(y), np.max(y)
            self.objects[i].img = np.repeat(img[y1:y2, x1:x2][...,None], self.channels, 2)

    def run(self):
        self.img[...] = 0

        boxes = np.zeros((len(self.objects), 5), dtype=np.float32)
        for i, object in enumerate(self.objects):
            x1, y1, x2, y2 = object.run()
            boxes[i] = np.array([x1, y1, x2, y2, object.class_id + self.label_offset])
            #draw in roi resized version of img
            thumbnail = cv2.resize(object.img, (x2-x1, y2-y1), cv2.INTER_LINEAR)
            self.img[y1:y2, x1:x2] = np.maximum(self.img[y1:y2, x1:x2], thumbnail)

        output = np.transpose(self.img, [2, 0, 1])
        return output, boxes


class MovingMnistDataset(toy.SquaresVideos):
    def __init__(self, batchsize=32, t=10, h=300, w=300, c=3,
                 normalize=False, max_stops=30, max_objects=3,
                 max_classes=3, train=True):
        self.train = train
        super(MovingMnistDataset, self).__init__(batchsize, t, h, w, c,
                                                normalize, max_stops, max_objects,
                                                max_classes, 'none', True)
        self.labelmap = [str(i) for i in range(10)]

    def build(self):
        num_sets = max(1, self.num_batches // self.max_consecutive_batches)
        self.animations = [[MovingMnistAnimation(self.time, self.height,
                                                self.width, self.channels, self.max_stops,
                                                 self.max_objects, self.train)
                           for _ in range(self.batchsize)]
                           for _ in range(num_sets)]




if __name__ == '__main__':
    import torch
    from core.utils.vis import boxarray_to_boxes, draw_bboxes, make_single_channel_display

    batchsize = 4
    dataset = MovingMnistDataset(t=10, c=3, h=256, w=256, batchsize=batchsize, train=False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, num_workers=1,
                                             shuffle=False, collate_fn=opts.video_collate_fn_with_reset_info, pin_memory=True)
    dataloader.dataset.reset()
    start = 0
    for (x, y, r) in dataloader:
        if r:
            print('Reset!')
        for t in range(len(y)):
            for j in range(dataset.batchsize):
                boxes = y[t][j].cpu()
                boxes = boxes.cpu().numpy().astype(np.int32)
                bboxes = boxarray_to_boxes(boxes[:, :4], boxes[:, 4]-1, dataset.labelmap)

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
                    img = np.zeros((256, 256, 3), dtype=np.uint8)

                img = draw_bboxes(img, bboxes)

                cv2.imshow('example'+str(j), img)
                key = cv2.waitKey(5)
                if key == 27:
                    exit()


