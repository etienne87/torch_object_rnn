from __future__ import print_function

import time as timer
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

import core.recurrent as rnn
import core.utils as utils
from core.box_coder import SSDBoxCoder
from core.ssd_loss import SSDLoss

from tensorboardX import SummaryWriter
from toy_pbm_detection import SquaresVideos, PrevNext


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup, momentum=0.01),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes, device=labels.device)  # [D,D]
    return y[labels]  # [N,D]


def prepare_fmap_input(box_coder, loc_targets, cls_targets, num_classes, batchsize):
    r"""Use fmap sizes to encode
    loc_targets: TxN, D, 4
    cls_targets: TxN, D, 1

    output packed T, N, (4+C), H, W to feed the network!
    """
    n_anchors = 2 * len(box_coder.aspect_ratios) + 2
    cls_targets = one_hot_embedding(cls_targets, num_classes)
    loc = torch.cat((loc_targets, cls_targets), dim=2)
    out = torch.split(loc, box_coder.fm_len, dim=1)
    res = []
    for fm_size, tensor in zip(box_coder.fm_sizes, out):
        fm_h, fm_w = fm_size
        n, l, c = tensor.size()
        assert l == fm_h * fm_w * n_anchors
        ans = tensor.view(n, fm_h, fm_w, n_anchors * c)
        ans = ans.permute([0, 3, 1, 2])
        ans = rnn.batch_to_time(ans, batchsize)
        res.append(ans)
    return res


def encode_boxes(targets, box_coder, cuda):
    loc_targets, cls_targets = [], []
    for t in range(len(targets)):
        for i in range(len(targets[t])):
            boxes, labels = targets[t][i][:, :-1], targets[t][i][:, -1]
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


def decode_boxes(box_map, num_classes, num_anchors):
    """
    box_map: T, N, C, H, W
    """
    fm_h, fm_w = box_map.shape[-2:]
    nboxes = fm_h * fm_w * num_anchors
    box_map = rnn.time_to_batch(box_map)
    # TN, C, H, W -> TN, H, W, C -> TN, HWNa, 4+Classes
    box_map = box_map.permute([0, 2, 3, 1]).contiguous().view(box_map.size(0), nboxes, 4 + num_classes)
    return box_map[..., :4], box_map[..., 4:]


def get_box_params(sources, h, w):
    image_size = float(min(h, w))
    steps = []
    box_sizes = []
    fm_sizes = []
    s_min, s_max = 0.1, 0.9
    m = float(len(sources))
    for k, src in enumerate(sources):
        h2, w2 = src
        fm_sizes.append((h2, w2))
        step_y, step_x = math.floor(float(h) / h2), math.floor(float(w) / w2)
        steps.append((step_y, step_x))
        s_k = s_min + (s_max - s_min) * k / m
        box_sizes.append(math.floor(s_k * image_size))
    s_k = s_min + (s_max - s_min)
    box_sizes.append(s_k * image_size)
    return fm_sizes, steps, box_sizes


class BoxTracker(nn.Module):
    """takes hidden from above and fuse gt to hidden state

    """

    def __init__(self, cin, cout, nanchors, num_classes=2, stride=2):
        super(BoxTracker, self).__init__()
        self.nanchors = nanchors
        self.num_classes = num_classes
        self.gtc = nanchors * (4 + num_classes)
        self.cout = cout
        self.x2h = rnn.SequenceWise(conv_bn(cin, 4 * cout, stride=stride))
        self.h2h = nn.Conv2d(cout, 4 * cout, kernel_size=3, stride=1, padding=1)
        self.gt2h = nn.Conv2d(self.gtc, 4 * cout, kernel_size=3, stride=1, padding=1)
        self.h2gt = nn.Conv2d(self.cout, self.gtc, kernel_size=3, stride=1, padding=1)
        self.reset()

    def preprocess_output(self, ht, spatial=False):
        """
        C: Nanchors x (4+Classes)
        ht: (N, C, H, W) -> (N, Nanchors, (4+Classes), H, W)

        -> split to:
        1. (N, Nanchors, 4, H, W)
        2. (N, Nanchors, Classes, H, W)


        """
        n, chn, h, w = ht.size()
        ht = ht.detach()
        tmp = ht.view(n, self.nanchors, 4 + self.num_classes, h, w)
        loc = tmp[:, :, :4, :, :]
        cls = F.softmax(tmp[:, :, 4:, :, :], dim=2)
        tmp = torch.cat((loc, cls), dim=2).view(n, chn, h, w)
        return tmp

    @staticmethod
    def lstm_update(prev_c, tmp):
        cc_i, cc_f, cc_o, cc_g = tmp.chunk(4, 1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c = f * prev_c + i * g
        h = o * torch.tanh(c)
        return h, c

    def forward(self, x, prev_gt, alpha=1, future=0):
        x = self.x2h(x)
        xseq = x.unbind(0)
        gtseq = prev_gt.unbind(0)

        ht = self.ht
        xt = None

        # init prev_h
        if self.prev_h is None:
            n, c, h, w = xseq[0].shape

            self.prev_h = torch.zeros((n, self.cout, h, w), dtype=torch.float, device=x.device)
            self.prev_c = torch.zeros((n, self.cout, h, w), dtype=torch.float, device=x.device)
        else:
            self.prev_h = self.prev_h.detach()
            self.prev_c = self.prev_c.detach()

        memory = []
        result = []
        # First treat sequence
        for t, (xt, gt) in enumerate(zip(xseq, gtseq)):
            v = random.uniform(0, 1)
            if ht is not None and v > alpha:
                gt = self.preprocess_output(ht.detach())

            tmp = self.gt2h(gt) + self.h2h(self.prev_h) + xt

            h, c = BoxTracker.lstm_update(self.prev_c, tmp)
            self.prev_h = h
            self.prev_c = c

            # actual prediction of boxes
            ht = self.h2gt(h)
            result.append(ht[None])
            memory.append(h[None])
            self.time += 1

        # For auto-regressive use-cases
        if future:
            for _ in range(future):
                gt = self.preprocess_output(ht.detach())

                tmp = self.gt2h(gt) + self.h2h(self.prev_h)  # no xt
                h, c = BoxTracker.lstm_update(self.prev_c, tmp)
                self.prev_h = h
                self.prev_c = c

                # actual prediction of boxes
                ht = self.h2gt(h)
                result.append(ht[None])
                memory.append(h[None])

        result = torch.cat(result, dim=0)
        memory = torch.cat(memory, dim=0)
        return memory, result

    def reset(self):
        self.prev_h, self.prev_c = None, None
        self.ht = None  # actual output
        self.time = 0


class ARSSD(nn.Module):
    def __init__(self, num_classes, height, width):
        super(ARSSD, self).__init__()

        base = 16
        self.cnn = rnn.SequenceWise(
            nn.Sequential(
                conv_bn(1, base, 2),
                conv_bn(base, base<<1, 2))
        )

        # here we simulate the image preliminary cnn
        sources = [(height / 8, width / 8),
                   (height / 16, width / 16),
                   (height / 32, width / 32)]

        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.fm_sizes, self.steps, self.box_sizes = get_box_params(sources, height, width)
        self.aspect_ratios = []
        self.num_anchors = 2 * len(self.aspect_ratios) + 2

        self.box_trackers = nn.ModuleList([
            BoxTracker(base<<1, base<<2, self.num_anchors, num_classes, stride=2),
            BoxTracker(base<<2, base<<3, self.num_anchors, num_classes, stride=2),
            BoxTracker(base<<3, base<<3, self.num_anchors, num_classes, stride=2)
        ])

    def forward(self, images, gts, alpha=1.0, future=0):
        """

        :param images: sequence of images
        :param xl: previous gt encoded using a box-to-image encoder
        :param alpha: probability to use the previous output as gt input
        :param future: hallucination steps will run every tracker without any image input
        :return:
        """
        x = self.cnn(images)
        hidden = None
        loc_preds, cls_preds = [], []
        for bt, gt in zip(self.box_trackers, gts):
            x, y = bt(x, gt, alpha, future)
            loc, cls = decode_boxes(y, self.num_classes, self.num_anchors)
            loc_preds.append(loc)
            cls_preds.append(cls)
        loc_preds = torch.cat(loc_preds, dim=1)
        cls_preds = torch.cat(cls_preds, dim=1)
        return loc_preds, cls_preds

    def reset(self):
        for i in range(len(self.box_trackers)):
            self.box_trackers[i].reset()


if __name__ == '__main__':
    num_classes, cin, tbins, height, width = 2, 1, 10, 128, 128
    batchsize = 8
    epochs = 100
    cuda = 1
    train_iter = 100
    nclasses = 2
    dataset = SquaresVideos(t=tbins, c=cin, h=height, w=width, mode='diff',
                            batchsize=batchsize, max_classes=num_classes - 1, render=True)
    dataset.num_frames = train_iter

    prevnext = PrevNext()

    conv_encoding = True

    net = ARSSD(num_classes, height, width)
    box_coder = SSDBoxCoder(net, 0.5)

    if cuda:
        net.cuda()
        box_coder.cuda()

    logdir = 'boxexp/random_alpha_0_during_test/'
    writer = SummaryWriter(logdir)

    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = SSDLoss(num_classes=2)

    proba = 0.5
    gt_every = 1
    alpha = 0.5

    for epoch in range(epochs):
        # train round
        print('TRAIN: PROBA RESET: ', proba, ' ALPHA: ', alpha)
        net.reset()
        for batch_idx, data in enumerate(dataset):
            if np.random.rand() < proba:
                dataset.reset()
                net.reset()

            images, targets = dataset.next()
            images, targets, prev_targets = prevnext(images, targets)


            optimizer.zero_grad()

            if cuda:
                images = images.cuda()

            loc_targets, cls_targets = encode_boxes(targets, box_coder, cuda)
            prev_loc_targets, prev_cls_targets = encode_boxes(prev_targets, box_coder, cuda)
            fmaps = prepare_fmap_input(box_coder, prev_loc_targets, prev_cls_targets, nclasses, batchsize)
            loc_preds, cls_preds = net(images, fmaps, alpha)

            loc_loss, cls_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss = loc_loss + cls_loss

            loss.backward()

            optimizer.step()
            if batch_idx % 3 == 0:
                print('iter: ', batch_idx, ' loss:', loss.item())
            writer.add_scalar('train_loss', loss.data.item(), batch_idx + epoch * len(dataset))

        # val round: make video, initialize from real seq
        print('DEMO:')
        with torch.no_grad():

            periods = 10
            nrows = 2
            ncols = batchsize / nrows
            net.reset()
            grid = np.ones((periods * tbins, nrows, ncols, dataset.height, dataset.width, 3), dtype=np.uint8) * 127

            for period in range(periods):
                images, targets = dataset.next()
                images, targets, prev_targets = prevnext(images, targets)


                if cuda:
                    images = images.cuda()

                prev_loc_targets, prev_cls_targets = encode_boxes(prev_targets, box_coder, cuda)
                fmaps = prepare_fmap_input(box_coder, prev_loc_targets, prev_cls_targets, nclasses, batchsize)
                loc_preds, cls_preds = net(images, fmaps, alpha=0)

                loc_preds = rnn.batch_to_time(loc_preds, batchsize).data
                cls_preds = F.softmax(rnn.batch_to_time(cls_preds, batchsize).data, dim=-1)
                images = images.cpu().data.numpy()

                for t in range(tbins):
                    for i in range(batchsize):
                        y = i / ncols
                        x = i % ncols
                        img = utils.general_frame_display(images[t, i])

                        boxes, labels, scores = box_coder.decode(loc_preds[t, i],
                                                                 cls_preds[t, i],
                                                                 nms_thresh=0.6,
                                                                 score_thresh=0.3)
                        if boxes is not None:
                            bboxes = utils.boxarray_to_boxes(boxes, labels, dataset.labelmap)
                            img = utils.draw_bboxes(img, bboxes)

                        grid[t + period * tbins, y, x] = img

            print('DEMO PART 1: DONE')
            grid = grid.swapaxes(2, 3).reshape(periods * tbins, nrows * dataset.height, ncols * dataset.width, 3)
            utils.add_video(writer, 'test_with_xt', grid, global_step=epoch, fps=30)

            grid = np.ones((periods * tbins, nrows, ncols, dataset.height, dataset.width, 3), dtype=np.uint8) * 127
            images, targets = dataset.next()
            images, targets, prev_targets = prevnext(images, targets)

            if cuda:
                images = images.cuda()
            prev_loc_targets, prev_cls_targets = encode_boxes(prev_targets, box_coder, cuda)
            fmaps = prepare_fmap_input(box_coder, prev_loc_targets, prev_cls_targets, nclasses, batchsize)
            loc_preds, cls_preds = net(images, fmaps, future=int(periods * tbins), alpha=0)
            loc_preds = rnn.batch_to_time(loc_preds, batchsize).data
            cls_preds = F.softmax(rnn.batch_to_time(cls_preds, batchsize).data, dim=-1)
            images = images.cpu().data.numpy()

            for t in range(periods * tbins):
                for i in range(batchsize):
                    y = i / ncols
                    x = i % ncols
                    if t < tbins:
                        img = utils.general_frame_display(images[t, i])
                    else:
                        img = grid[t, y, x]

                    boxes, labels, scores = box_coder.decode(loc_preds[t, i],
                                                             cls_preds[t, i],
                                                             nms_thresh=0.6,
                                                             score_thresh=0.3)
                    if boxes is not None:
                        color = (0, 255, 0) if t <= tbins else (255, 0, 0)

                        bboxes = utils.boxarray_to_boxes(boxes, labels, dataset.labelmap)
                        img = utils.draw_bboxes(img, bboxes, color)

                    grid[t, y, x] = img

            grid = grid.swapaxes(2, 3).reshape(periods * tbins, nrows * dataset.height, ncols * dataset.width, 3)
            utils.add_video(writer, 'test_hallucinate', grid, global_step=epoch, fps=30)

            print('DEMO PART 2: DONE')

        scheduler.step()
        proba *= 0.9
        # alpha *= 0.9
        # alpha = max(0.1, alpha)