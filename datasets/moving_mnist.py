from __future__ import print_function
from __future__ import absolute_import
from __future__ import division 


import numpy as np
import random
import cv2
import datasets.moving_box_detection as toy
from torchvision import datasets, transforms

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
    def __init__(self, t=20, h=128, w=128, c=3, max_stop=15,
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
            thumbnail = cv2.resize(object.img, (x2-x1, y2-y1), cv2.INTER_LINEAR)
            self.img[y1:y2, x1:x2] = np.maximum(self.img[y1:y2, x1:x2], thumbnail)
        output = self.img * 255
        return output, boxes


class MnistEnv(object):
    def __init__(self, num=3, niter=1000):
        self.envs = [MovingMnistAnimation(t=10, h=128, w=128, c=3) for i in range(num)]
        self.niter = niter
        self.reset() 
        self.max_steps = 50
        self.step = 0
        self.max_rounds = niter

    def reset(self):
        for env in self.envs:
            env.reset()

    def next(self, arrays):
        tbins = arrays.shape[1]
        all_boxes = []
        reset = self.step > self.max_steps
        if reset: 
            self.step
            for env in self.envs:
                env.reset()    
        for i, env in enumerate(self.envs):
            env_boxes = []
            for t in range(tbins):
                observation, boxes = env.run()  
                arrays[i, t] = observation   
                env_boxes.append(boxes)
            all_boxes.append(env_boxes)

        self.step += tbins
        return {'boxes': all_boxes, 'reset': [reset]*len(self.envs)}



if __name__ == '__main__':
    import sys
    import time
    from datasets.multistreamer import MultiStreamer
    from core.utils.vis import boxarray_to_boxes, draw_bboxes, make_single_channel_display

    labelmap = [str(i) for i in range(10)]
    show_batchsize = 4
    tbins, height, width, cin = 10, 128, 128, 3
    array_dim = (tbins, height, width, cin)
    dataloader = MultiStreamer(MnistEnv, array_dim, batchsize=16, max_q_size=4, num_threads=4)

    start = 0

    nrows = 2 ** ((show_batchsize.bit_length() - 1) // 2)
    ncols = show_batchsize // nrows

    grid = np.zeros((nrows, ncols, height, width, 3), dtype=np.uint8)

    for i, data in enumerate(dataloader):
        batch, targets = data['data'], data['boxes']
        runtime = time.time() - start
        for t in range(tbins):
            grid[...] = 0
            for n in range(show_batchsize):
                img = batch[n, t]

                boxes = targets[n][t] 
                boxes = boxes.astype(np.int32)
                bboxes = boxarray_to_boxes(boxes[:, :4], boxes[:, 4]-1, labelmap)
                img = draw_bboxes(img, bboxes)

                
                grid[n//ncols, n%ncols] = img
            im = grid.swapaxes(1, 2).reshape(nrows * height, ncols * width, 3)
            cv2.imshow('dataset', im)
            key = cv2.waitKey(5)
            if key == 27:
                break
        
        
        sys.stdout.write('\rtime: %f' % (runtime))
        sys.stdout.flush()
        start = time.time()
