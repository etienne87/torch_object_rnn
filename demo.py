from __future__ import print_function

import os
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from ssd_v2 import SSD300
from box_coder import SSDBoxCoder

import sys

sys.path.insert(0, '../')
from data import krvideo
from h5_sequence_detection import H5SequenceDetection

import cv2
import math
import time

parser = argparse.ArgumentParser(description='PyTorch SSD Demo')
parser.add_argument('--video', type=str, help='path to test video')
parser.add_argument('--checkpoint', default='/home/eperot/workspace/data/torch_ssd/ckpt.pth', type=str,
                    help='checkpoint path')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--gpu_dev', type=int, default=0, help='gpu number to use')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_dev)

from toy_pbm_detection import WhiteSquaresImages, WhiteSquaresVideos
from h5_sequence_detection import H5SequenceDetection
from utils import draw_bboxes, make_single_channel_display, filter_outliers

CLASSES, CIN, T, H, W = 3, 2, 10, 120, 152

# Model
print('==> Building model..')
net = SSD300(num_classes=CLASSES, in_channels=CIN, height=H, width=W)

if args.cuda:
    net.cuda()
    cudnn.benchmark = True

print('==> Resuming from checkpoint..')
checkpoint = torch.load(args.checkpoint)
net.load_state_dict(checkpoint['net'])
best_loss = checkpoint['loss']
start_epoch = checkpoint['epoch']

box_coder = SSDBoxCoder(net)

if args.cuda:
    box_coder.cuda()

img_size = (120, 152)
labelmap = ['car', 'pedestrian']

def histogram(events, bins, cuda):
    i = torch.from_numpy(events).long() # "sending the events on the gpu"
    size = torch.Size(bins.tolist())
    if cuda > 0:
        i = i.cuda(0) #indices
        v = torch.cuda.FloatTensor(i.size(0)).fill_(1.0)  # values
        hist = torch.cuda.sparse.FloatTensor(i.t(), v, size).to_dense()  # happens on the gpu
    else:
        v = torch.FloatTensor(i.size(0)).fill_(1.0)
        hist = torch.sparse.FloatTensor(i.t(), v, size).to_dense()  # happens on the gpu

    return hist

def _dig_out_time(x, n=1):
    nt, nanchors, c = x.size()
    t = int(nt / n)
    x = x.view(n, t, nanchors, c)
    return x

def boxarray_to_boxes(boxes, labels, labelmap):
    boxes = boxes.cpu().numpy().astype(np.int32)
    labels = labels.cpu().numpy().astype(np.int32)
    bboxes = []
    for label,box in zip(labels,boxes):
        class_name = labelmap[label]
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        bb = (class_name, label, pt1, pt2, None, None, None)
        bboxes.append(bb)
    return bboxes

KRVIDEO = True
def demo():
    stop = False
    tmax = 50000
    ybins, xbins = 120, 152

    bins = np.array([2, 1, ybins, xbins])

    if KRVIDEO:
        #path = '/home/eperot/waymo/left_drive_1_master_td.dat'
        path = os.environ['DATASET_DIR'] + '/recordings/2017/2017_04_06-Paris_to_Hyere/paris_to_hyere_17-04-06_09-57-37_td.dat'
        #path = os.environ['DATASET_DIR'] + '/recordings/2017/2017_03_30-test_chrono_car/recording_17-03-30_11-41-02_td.dat'
        video = krvideo.ChronoVideo(path, timestep=tmax)
        H, W = video.height, video.width
        edges = np.array([2, tmax, H, W], dtype=np.float32)
        divider = np.round(edges / bins).astype(int)

    else:
        H, W = ybins, xbins
        edges = np.array([2, tmax, H, W], dtype=np.float32)
        divider = np.round(edges / bins).astype(int)
        video = H5SequenceDetection('/media/eperot/Elements/torch_dataset/val.h5', batchsize=1, tbins=10, ybins=H, xbins=W, cuda=args.cuda)


    net.eval()
    net.reset()

    while 1:#not video.done():
        if KRVIDEO:
            if not stop:
                try:
                    x, y, p, t = video.load_delta_t(video.current_time() + tmax)
                except:
                    break
            if not len(t): continue

            # Histogram Preparation
            events = np.vstack((p, t, y, x)).T.astype(np.int)
            events /= divider
            events = np.minimum(events, bins - 1)
            input = histogram(events, bins, args.cuda)
            input = torch.clamp(input, 0, 32)/32.0
            input = input.unsqueeze(0)
        else:
            input, _ = video.next()

        # Network Inference
        start = time.time()
        loc_preds, cls_preds = net(input)
        print(time.time()-start, ' s')


        loc_preds = _dig_out_time(loc_preds, 1)
        cls_preds = _dig_out_time(cls_preds, 1)



        img = input.cpu().numpy()[0]

        diff = filter_outliers(img[1,-1] - img[0,-1])
        img = make_single_channel_display(diff)

        boxes, labels, scores = box_coder.decode(loc_preds[0,-1].data, F.softmax(cls_preds[0,-1], dim=1).data)
        if boxes is not None:
            bboxes = boxarray_to_boxes(boxes, labels, labelmap)
            img = draw_bboxes(img, bboxes)


        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        cv2.imshow('img', img)
        key = cv2.waitKey(20)
        if key & 0xFF == ord('p'):
            stop = 1 - stop


def main():
    demo()


if __name__ == '__main__':
    main()