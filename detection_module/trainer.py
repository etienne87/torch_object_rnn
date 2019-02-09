from __future__ import print_function
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tensorboardX import SummaryWriter
from utils import draw_bboxes, make_single_channel_display


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
def _encode_boxes(targets, box_coder, cuda, all_timesteps=True):
    loc_targets, cls_targets = [], []

    if all_timesteps:
        for t in range(len(targets)):
            for i in range(len(targets[t])):
                boxes, labels = targets[t][i][:, :-1], targets[t][i][:, -1]
                if cuda:
                    boxes, labels = boxes.cuda(), labels.cuda()
                loc_t, cls_t = box_coder.encode(boxes, labels)
                loc_targets.append(loc_t.unsqueeze(0))
                cls_targets.append(cls_t.unsqueeze(0).long())
    else:
        if isinstance(targets[0], list):
            targets = [item[-1] for item in targets] #take last item
        for i in range(len(targets)):
            boxes, labels = targets[i][:, :-1], targets[i][:, -1]
            if cuda:
                boxes, labels = boxes.cuda(), labels.cuda()
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
    x = x.view(t, n, nanchors, c)
    return x


def single_frame_display(im):
    return make_single_channel_display(im[0], -1, 1)


def prepare_ckpt_dir(filename):
    dir = os.path.dirname(filename)
    if not os.path.isdir(dir):
        os.mkdir(dir)


class SSDTrainer(object):
    """
    class wrapping training/ validation/ testing
    """
    def __init__(self, net, box_coder, criterion, optimizer, all_timesteps=False):
        self.net = net
        self.box_coder = box_coder
        self.criterion = criterion
        self.optimizer = optimizer
        self.all_timesteps = all_timesteps
        self.make_image = single_frame_display
        self.writer = SummaryWriter()

    def __del__(self):
        self.writer.close()

    def train(self, epoch, dataset, args):
        print('\nEpoch: %d (train)' % epoch)
        self.net.train()
        self.net.reset()
        train_loss = 0
        self.net.extractor.return_all = self.all_timesteps

        start = 0
        runtime_stats = {'dataloader': 0, 'network': 0}
        for batch_idx, data in enumerate(dataset):
            if batch_idx > 0:
                runtime_stats['dataloader'] += time.time()-start
            inputs, targets = data

            start = time.time()
            if args.cuda:
                inputs = inputs.cuda()

            self.optimizer.zero_grad()
            loc_preds, cls_preds = self.net(inputs)
            loc_targets, cls_targets = _encode_boxes(targets, self.box_coder, args.cuda, self.all_timesteps)

            loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss = loc_loss + cls_loss
            loss.backward()
            self.optimizer.step()

            runtime_stats['network'] += time.time() - start

            train_loss += loss.data.item()

            self.writer.add_scalar('train_loss',loss.data.item(), batch_idx + epoch * len(dataset))

            if batch_idx % args.log_every == 0:
                print('\rtrain_loss: %.3f | avg_loss: %.3f [%d/%d] | @data: %.3f | @net: %.3f'
                      % (loss.data.item(), train_loss / (batch_idx + 1), batch_idx + 1, len(dataset),
                         runtime_stats['dataloader'] / (batch_idx + 1),
                         runtime_stats['network'] / (batch_idx + 1)
                         ), ' ')

            start = time.time()


    def val(self, epoch, dataset, args):
        print('\nEpoch: %d (val)' % epoch)
        self.net.eval()
        self.net.reset()
        val_loss = 0

        for batch_idx, data in enumerate(dataset):
            inputs, targets = data
            if args.cuda:
                inputs = inputs.cuda()
            loc_preds, cls_preds = self.net(inputs)
            loc_targets, cls_targets = _encode_boxes(targets, self.box_coder, args.cuda, self.all_timesteps)
            loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss = loc_loss + cls_loss
            val_loss += loss.item()

        self.writer.add_scalar('val_loss', loss.data.item(), epoch)

    def test(self, epoch, dataset, nrows, args):
        print('\nEpoch: %d (test)' % epoch)
        self.net.eval()
        self.net.reset()
        self.net.extractor.return_all = True

        periods = args.test_iter
        batchsize = dataset.batchsize
        time = dataset.time

        ncols = batchsize / nrows
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        videofile = os.path.dirname(args.checkpoint) + 'tests/' + str(epoch) + '.avi'
        prepare_ckpt_dir(videofile)
        out = cv2.VideoWriter(videofile, fourcc, 30.0, (ncols * dataset.height, nrows * dataset.width))
        grid = np.zeros((nrows, ncols, dataset.height, dataset.width, 3), dtype=np.uint8)

        #grid = torch.ByteTensor(dataset.batchsize, periods * time, 3, dataset.height, dataset.width)
        for period in range(periods):
            inputs, targets = dataset.next()

            if args.cuda:
                inputs = inputs.cuda()

            loc_preds, cls_preds = self.net(inputs)

            images = inputs.cpu().data.numpy()
            loc_preds = _dig_out_time(loc_preds, dataset.batchsize)
            cls_preds = _dig_out_time(cls_preds, dataset.batchsize)
            for t in range(loc_preds.size(0)):
                for i in range(loc_preds.size(1)):
                    y = i / ncols
                    x = i % ncols
                    img = self.make_image(images[t, i])
                    # assert img.shape == grid[0, 0].shape
                    boxes, labels, scores = self.box_coder.decode(loc_preds[t, i].data,
                                                                  F.softmax(cls_preds[t, i], dim=1).data,
                                                                  nms_thresh=0.6)
                    if boxes is not None:
                        bboxes = boxarray_to_boxes(boxes, labels, dataset.labelmap)
                        img = draw_bboxes(img, bboxes)

                    grid[y, x] = img
                    # grid[i, t + period * time] = torch.from_numpy(np.transpose(img, (2, 0, 1)))
                image = grid.swapaxes(1, 2).reshape(nrows * dataset.height, ncols * dataset.width, 3)
                out.write(image)

        out.release()

        #self.writer.add_video('test', vid_tensor=grid, fps=30)
        self.net.extractor.return_all = False

    def save_ckpt(self, epoch, args, name='checkpoint#'):
        state = {
            'net': self.net.state_dict(),
            'epoch': epoch,
        }
        ckpt_file = args.checkpoint + '/' + name + str(epoch) + '.pth'
        prepare_ckpt_dir(ckpt_file)
        torch.save(state, ckpt_file)