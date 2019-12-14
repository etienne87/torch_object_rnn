from __future__ import print_function
from __future__ import absolute_import
from __future__ import division 

import argparse 
import sys
import time

import torch
import numpy as np
import random
import cv2
import datasets.moving_box_detection as toy
from datasets.multistreamer import MultiStreamer

from torchvision import datasets, transforms
from functools import partial 


from core.utils.vis import boxarray_to_boxes, draw_bboxes, make_single_channel_display


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
                max_objects=2, proc_id = 0, train=True):
        self.dataset_ = TRAIN_DATASET if train else TEST_DATASET
        self.label_offset = 1
        self.proc_id = proc_id
        np.random.seed(proc_id)
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
        output = self.img 
        return output, boxes


class MnistEnv(object):
    def __init__(self, proc_id=0, num_procs=1, num_envs=3, niter=100, **kwargs):
        self.envs = [MovingMnistAnimation(proc_id=proc_id+i, **kwargs) for i in range(num_envs)]
        self.niter = niter
        self.reset() 
        self.max_steps = 500
        self.step = 0
        self.max_iter = niter
        self.proc_id = proc_id
        self.labelmap = [str(i) for i in range(10)]
        self.label_offset = 1

    def reset(self):
        for env in self.envs:
            env.reset()

    def next(self, arrays):
        tbins = arrays.shape[1]
        all_boxes = []
        reset = self.step > self.max_steps
        if reset: 
            self.step = 0
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
        return {'boxes': all_boxes, 'resets': [reset]*len(self.envs)}


def collate_fn(data):
    #permute NTCHW - TNCHW
    batch, boxes, resets = data['data'], data['boxes'], data['resets']
    batch = torch.from_numpy(batch).permute(1,0,4,2,3)
    t, n = batch.shape[:2]
    boxes = [[torch.from_numpy(boxes[i][t]) for i in range(n)] for t in range(t)]
    resets = 1-torch.FloatTensor(resets)
    resets = resets[:,None,None,None]
    return {'data': batch, 'boxes': boxes, 'resets': resets}


def make_moving_mnist(args):
    tbins, height, width, cin = 10, 256, 256, 3
    array_dim = (tbins, height, width, cin)
    env_train = partial(MnistEnv, niter=args.train_iter, t=tbins, h=height, w=width, c=cin, train=True)
    env_val = partial(MnistEnv, niter=args.test_iter, t=tbins, h=height, w=width, c=cin, train=False)
    train_dataset = MultiStreamer(env_train, array_dim, batchsize=args.batchsize, max_q_size=4, num_threads=args.num_workers, collate_fn=collate_fn)
    test_dataset = MultiStreamer(env_val, array_dim, batchsize=args.batchsize, max_q_size=4, num_threads=args.num_workers, collate_fn=collate_fn)
    classes = 10
    return train_dataset, test_dataset, classes





def parse_args():
    parser = argparse.ArgumentParser(description='Mnist Reader')
    parser.add_argument('--batchsize', type=int, default=8, help='batchsize')
    parser.add_argument('--num_workers', action='store_true', help="viz videos or images")
    return parser.parse_args()


if __name__ == '__main__':
   
    args = parse_args()
    dataloader, _, _ = make_moving_mnist(args)
    show_batchsize = dataloader.batchsize

    start = 0

    nrows = 2 ** ((show_batchsize.bit_length() - 1) // 2)
    ncols = show_batchsize // nrows

    grid = np.zeros((nrows, ncols, 256, 256, 3), dtype=np.uint8)

    for i, data in enumerate(dataloader):
        batch, targets = data['data'], data['boxes']
        height, width = batch.shape[-2], batch.shape[-1]
        runtime = time.time() - start
        for t in range(10):
            grid[...] = 0
            for n in range(dataloader.batchsize):
                img = batch[t,n].permute(1, 2, 0).cpu().numpy()*255
                boxes = targets[t][n].numpy() 
                boxes = boxes.astype(np.int32)
                bboxes = boxarray_to_boxes(boxes[:, :4], boxes[:, 4]-1, dataloader.dataset.labelmap)
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
