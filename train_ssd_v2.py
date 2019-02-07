from __future__ import print_function

import os
import time as timing
import argparse
import numpy as np
import cv2

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from ssd_v2 import SSD
from box_coder import SSDBoxCoder
from ssd_loss import SSDLoss

from toy_pbm_detection import SquaresImages, SquaresVideos
from utils import draw_bboxes, make_single_channel_display, filter_outliers


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SSD Training')
    parser.add_argument('--path', type=str, default='', help='path to dataset')
    parser.add_argument('--batchsize', type=int, default=32, help='batchsize')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--train_iter', type=int, default=200, help='#iter / train epoch')
    parser.add_argument('--test_iter', type=int, default=200, help='#iter / test epoch')
    parser.add_argument('--epochs', type=int, default=1000, help='num epochs to train')
    parser.add_argument('--model', default='./checkpoints/ssd300_v2.pth', type=str, help='initialized model path')
    parser.add_argument('--checkpoint', default='./checkpoints/ckpt.pth', type=str, help='checkpoint path')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--log_every', type=int, default=10, help='log every')
    return parser.parse_args()


def boxarray_to_boxes(boxes, labels, labelmap):
    boxes = boxes.cpu().numpy().astype(np.int32)
    labels = labels.cpu().numpy().astype(np.int32)
    bboxes = []
    for label, box in zip(labels, boxes):
        class_name = labelmap[label]
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        bb = (class_name, label, pt1, pt2, None, None, None)
        bboxes.append(bb)
    return bboxes


# Training
def _encode_boxes(targets, box_coder, cuda):
    loc_targets, cls_targets = [], []
    for i in range(len(targets)):
        boxes, labels = targets[i][:, :-1], targets[i][:, -1]
        loc_t, cls_t = box_coder.encode(boxes, labels)
        loc_targets.append(loc_t.unsqueeze(0))
        cls_targets.append(cls_t.unsqueeze(0).long())

    loc_targets = torch.cat(loc_targets, dim=0)  # (N,#anchors,4)
    cls_targets = torch.cat(cls_targets, dim=0)  # (N,#anchors,C)

    if cuda:
        loc_targets = loc_targets.cuda()
        cls_targets = cls_targets.cuda()

    return loc_targets, cls_targets


def train(epoch, net,  box_coder, criterion, dataset, optimizer, args):
    print('\nEpoch: %d' % epoch)
    net.train()
    net.reset()
    train_loss = 0

    for batch_idx in range(args.train_iter):
        inputs, targets = dataset.next()

        if args.cuda:
            inputs = inputs.cuda()
            targets = [y.cuda() for y in targets]

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)

        loc_targets, cls_targets = _encode_boxes(targets, box_coder, args.cuda)

        loc_loss, cls_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss = loc_loss + cls_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()

        if batch_idx % args.log_every == 0:
            print('\rtrain_loss: %.3f | avg_loss: %.3f [%d/%d]'
                  % (loss.data.item(), train_loss / (batch_idx + 1), batch_idx + 1, len(dataset)), ' ')


def _dig_out_time(x, n=32):
    nt, nanchors, c = x.size()
    t = int(nt / n)
    x = x.view(n, t, nanchors, c)
    return x


def test(epoch, net, box_coder, dataset, nrows, args):
    print('\nEpoch (test): %d' % epoch)
    net.eval()
    net.reset()

    if isinstance(dataset, SquaresVideos):

        periods = args.test_iter

        batchsize = dataset.batchsize
        time = dataset.time

        ncols = batchsize / nrows

        grid = np.zeros((time * periods, nrows, ncols, dataset.height, dataset.width, 3), dtype=np.uint8)
        net.reset()
        net.extractor.return_all = True

        for period in range(periods):
            # print('\rperiod: ', period, end='')
            inputs, _ = dataset.next()
            images = inputs.cpu().data.numpy()

            if args.cuda:
                inputs = inputs.cuda()

            # start = timing.time()
            loc_preds, cls_preds = net(inputs)
            # end = timing.time()
            # print(end-start, ' s ')

            loc_preds = _dig_out_time(loc_preds, batchsize)
            cls_preds = _dig_out_time(cls_preds, batchsize)

            for i in range(loc_preds.size(0)):
                y = i / ncols
                x = i % ncols
                for t in range(loc_preds.size(1)):
                    if dataset.channels == 3:
                        img = np.moveaxis(images[i, :, t], 0, 2)
                        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                        show = np.zeros((dataset.height, dataset.width, 3), dtype=np.float32)
                        show[...] = img
                        img = show
                    elif dataset.channels == 2:
                        diff = filter_outliers(images[i, 1, t] - images[i, 0, t])
                        img = make_single_channel_display(diff, None, None)
                    else:
                        img = make_single_channel_display(images[i, 0, t], -1, 1)
                    boxes, labels, scores = box_coder.decode(loc_preds[i, t].data,
                                                             F.softmax(cls_preds[i, t], dim=1).data, nms_thresh=0.6)
                    if boxes is not None:
                        bboxes = boxarray_to_boxes(boxes, labels, dataset.labelmap)
                        img = draw_bboxes(img, bboxes)

                    grid[t + period * time, y, x] = img

        video = grid.swapaxes(2, 3)  # (T,Rows,Cols,H,W,3) -> (T,Rows,H,Cols,W,3)
        video = video.reshape(time * periods, nrows * dataset.height, ncols * dataset.width,
                              3)  # (T,Rows,H,Cols,W,3) -> (T,Rows*H,Cols*W,3)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        videofile = './checkpoints/tests/' + str(epoch) + '.avi'
        print('writing to: ', videofile)
        out = cv2.VideoWriter(videofile, fourcc, 30.0, (ncols * dataset.height, nrows * dataset.width))
        for t in range(time * periods):
            out.write(video[t])
        out.release()

        net.extractor.return_all = False


def save_ckpt(epoch, net, args, name='ckpt'):
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    ckpt_file = os.path.dirname(args.checkpoint) + '/ckpt_' + str(epoch) + '.pth'
    if not os.path.isdir(os.path.dirname(args.checkpoint)):
        os.mkdir(os.path.dirname(args.checkpoint))
    torch.save(state, ckpt_file)


def main():
    args = parse_args()

    classes, cin, time, height, width = 2, 1, 5, 128, 128

    nrows = 4

    # Dataset
    print('==> Preparing dataset..')
    dataset = SquaresVideos(t=time, c=cin, h=height, w=width, batchsize=args.batchsize, normalize=False, cuda=args.cuda)


    # Model
    print('==> Building model..')
    net = SSD(num_classes=classes, cin=cin, height=height, width=width)

    if args.cuda:
        net.cuda()
        cudnn.benchmark = True

    best_loss = float('inf')  # best test loss
    start_epoch = 0  # start from epoch 0 or last epoch
    if args.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']

    box_coder = SSDBoxCoder(net)

    if args.cuda:
        box_coder.cuda()

    img_size = (dataset.height, dataset.width)

    criterion = SSDLoss(num_classes=classes)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch, net, box_coder, criterion, dataset, optimizer, args)
        test(epoch, net, box_coder, dataset, nrows, args)
        # scheduler.step()
        save_ckpt(epoch)



if __name__ == '__main__':
    main()