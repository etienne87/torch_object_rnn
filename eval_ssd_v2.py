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
from data import label as labelIO

import math

parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--dbname', type=str, default='/home/eperot/data/torch_dataset/train.h5', help='path to test db')
parser.add_argument('--output_path', type=str, default='/home/eperot/ssd/results/', help='path to output directory')
parser.add_argument('--batchsize', type=int, default=32, help='batchsize')
parser.add_argument('--max_iter', type=int, default=10, help='#iter / train epoch')
parser.add_argument('--checkpoint', default='/home/eperot/ssd/model/checkpoint/ckpt.pth', type=str,
                    help='checkpoint path')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--gpu_dev', type=int, default=0, help='gpu number to use')
parser.add_argument('--port', type=int, default=8008, help='visdom port')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_dev)

from toy_pbm_detection import WhiteSquaresImages, WhiteSquaresVideos
from h5_sequence_detection import H5SequenceDetection
from utils import draw_bboxes, make_single_channel_display

CLASSES, CIN, T, H, W = 3, 2, 10, 120, 152

nrows = 4

# Dataset
print('==> Preparing dataset..')
dataset = H5SequenceDetection(args.dbname, batchsize=args.batchsize, tbins=10, ybins=H, xbins=W, cuda=args.cuda)

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

img_size = (dataset.height, dataset.width)


# Dumps text files for evaluation
def eval(max_iter=500):
    net.eval()
    net.reset()

    output_labels_path = os.path.join(args.output_path, 'detections_labels_pos.txt')
    output_gt_path = os.path.join(args.output_path, 'gt_labels_pos.txt')
    gt_detections = {}
    output_detections = {}
    tf_to_kaer_map = {1: 7, 2: 15}
    min_diagonal = 5
    object_id = 0
    object_id_gt = 0
    # num_batches = dataset.count()
    # num_batches /= args.batchsize
    num_batches = 570
    print('TODO: ', num_batches)

    for batch_idx in range(num_batches):
        print(batch_idx)
        inputs, targets = dataset.next()

        if args.cuda:
            inputs = inputs.cuda()

        loc_preds, cls_preds = net(inputs)
        loc_targets, cls_targets = [], []
        for i in range(inputs.size(0)):
            boxes, labels = targets[i][:, :-1], targets[i][:, -1]
            loc_t, cls_t = box_coder.encode(boxes, labels)
            loc_targets.append(loc_t.unsqueeze(0))
            cls_targets.append(cls_t.unsqueeze(0).long())

        loc_targets = torch.cat(loc_targets, dim=0)  # (N,#anchors,4)
        cls_targets = torch.cat(cls_targets, dim=0)  # (N,#anchors,C)

        if args.cuda:
            loc_targets = loc_targets.cuda()
            cls_targets = cls_targets.cuda()

        for i in range(args.batchsize):
            current_ts = batch_idx * args.batchsize + i

            boxes, labels, scores = box_coder.decode(loc_preds[i].data, F.softmax(cls_preds[i], dim=1).data)
            if boxes is not None:
                output_detections[current_ts] = {}
                boxes = boxes.cpu().numpy().astype(np.int32)
                labels = labels.cpu().numpy().astype(np.int32)
                for label, box, score in zip(labels, boxes, scores):
                    x1, y1, w, h = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])

                    diag = math.sqrt(w ** 2 + h ** 2)
                    if diag <= min_diagonal:
                        continue

                    output_detections[current_ts][object_id] = {}
                    output_detections[current_ts][object_id]["class_id"] = tf_to_kaer_map[label + 1]
                    output_detections[current_ts][object_id]["bbox"] = [x1, y1, w, h]
                    output_detections[current_ts][object_id]["probability"] = score
                    object_id += 1

            gt_detections[current_ts] = {}
            boxes = targets[i].numpy()
            for box in boxes:
                x1, y1, w, h, label = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1]), int(box[4])
                gt_detections[current_ts][object_id_gt] = {}
                gt_detections[current_ts][object_id_gt]["class_id"] = tf_to_kaer_map[label + 1]
                gt_detections[current_ts][object_id_gt]["bbox"] = [x1, y1, w, h]
                gt_detections[current_ts][object_id_gt]["probability"] = 1.0
                object_id_gt += 1

    labelIO.write_bboxes(output_labels_path, output_detections, redundant=True, full_protocol=False,
                         tracked=False, delta_t=100000, default_class_id=0)

    labelIO.write_bboxes(output_gt_path, gt_detections, redundant=True, full_protocol=False,
                         tracked=False, delta_t=100000, default_class_id=0)


def _dig_out_time(x, n=32):
    nt, nanchors, c = x.size()
    t = int(nt / n)
    x = x.view(n, t, nanchors, c)
    return x


def main():
    eval(args.max_iter)


if __name__ == '__main__':
    main()