from __future__ import print_function
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from utils import draw_bboxes, make_single_channel_display, filter_outliers
from toy_pbm_detection import SquaresVideos


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

# batch to time for rank 3 tensors
def _dig_out_time(x, n=32):
    nt, nanchors, c = x.size()
    t = int(nt / n)
    x = x.view(n, t, nanchors, c)
    return x


class SSDTrainer(object):
    """
    class wrapping training/ validation/ testing
    """
    def __init__(self, net, box_coder, criterion, optimizer):
        self.net = net
        self.box_coder = box_coder
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, epoch, dataset, args):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        self.net.reset()
        train_loss = 0

        for batch_idx in range(args.train_iter):
            inputs, targets = dataset.next()

            if args.cuda:
                inputs = inputs.cuda()
                targets = [y.cuda() for y in targets]

            self.optimizer.zero_grad()
            loc_preds, cls_preds = self.net(inputs)

            loc_targets, cls_targets = _encode_boxes(targets, self.box_coder, args.cuda)

            loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss = loc_loss + cls_loss
            loss.backward()
            self.optimizer.step()

            train_loss += loss.data.item()

            if batch_idx % args.log_every == 0:
                print('\rtrain_loss: %.3f | avg_loss: %.3f [%d/%d]'
                      % (loss.data.item(), train_loss / (batch_idx + 1), batch_idx + 1, len(dataset)), ' ')


    def test(self, epoch, dataset, nrows, args):
        print('\nEpoch (test): %d' % epoch)
        self.net.eval()
        self.net.reset()

        if isinstance(dataset, SquaresVideos):

            periods = args.test_iter

            batchsize = dataset.batchsize
            time = dataset.time

            ncols = batchsize / nrows

            grid = np.zeros((time * periods, nrows, ncols, dataset.height, dataset.width, 3), dtype=np.uint8)
            self.net.reset()
            self.net.extractor.return_all = True

            for period in range(periods):
                # print('\rperiod: ', period, end='')
                inputs, _ = dataset.next()
                images = inputs.cpu().data.numpy()

                if args.cuda:
                    inputs = inputs.cuda()

                # start = timing.time()
                loc_preds, cls_preds = self.net(inputs)
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
                        boxes, labels, scores = self.box_coder.decode(loc_preds[i, t].data,
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

            self.net.extractor.return_all = False


    def save_ckpt(self, epoch, net, args, name='ckpt'):
        state = {
            'net': self.net.state_dict(),
            'epoch': epoch,
        }
        ckpt_file = os.path.dirname(args.checkpoint) + '/ckpt_' + str(epoch) + '.pth'
        if not os.path.isdir(os.path.dirname(args.checkpoint)):
            os.mkdir(os.path.dirname(args.checkpoint))
        torch.save(state, ckpt_file)