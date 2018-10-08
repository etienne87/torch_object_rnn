from __future__ import print_function

import os
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from ssd_v2 import SSD300
from box_coder import SSDBoxCoder
from ssd_loss import SSDLoss

import visdom
import cv2
import time as timing

parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--dbname', type=str, default='/home/eperot/data/torch_dataset/train.h5', help='path to db')
parser.add_argument('--batchsize', type=int, default=32, help='batchsize')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--max_iter', type=int, default=100, help='#iter / train epoch')
parser.add_argument('--test_iter', type=int, default=50, help='#iter / test epoch')
parser.add_argument('--epochs', type=int, default=1000, help='num epochs to train')
parser.add_argument('--model', default='./checkpoints/ssd300_v2.pth', type=str, help='initialized model path')
parser.add_argument('--checkpoint', default='./checkpoints/ckpt.pth', type=str,
                    help='checkpoint path')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--gpu_dev', type=int, default=0, help='gpu number to use')
parser.add_argument('--visdom', action='store_true', help='use visdom')
parser.add_argument('--port', type=int, default=8097, help='visdom port')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_dev)

from toy_pbm_detection import SquaresImages, SquaresVideos
from h5_sequence_detection import H5SequenceDetection
from utils import draw_bboxes, make_single_channel_display, filter_outliers


def boxarray_to_boxes(boxes, labels, labelmap):
    boxes = boxes.cpu().numpy().astype(np.int32)
    labels = labels.cpu().numpy().astype(np.int32)
    bboxes = []
    for label, box in zip(labels, boxes):
        class_name = dataset.labelmap[label]
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        bb = (class_name, label, pt1, pt2, None, None, None)
        bboxes.append(bb)
    return bboxes


CLASSES, CIN, T, H, W = 2, 3, 1, 120, 152

# H, W = H * 2, W * 2

nrows = 4

# Dataset
print('==> Preparing dataset..')
# dataset = SquaresImages(h=H, w=W, batchsize=args.batchsize, normalize=False, cuda=args.cuda)
dataset = SquaresVideos(h=H, w=W, batchsize=args.batchsize, normalize=False, cuda=args.cuda)
# dataset = H5SequenceDetection(args.dbname, batchsize=args.batchsize, tbins=10, ybins=H, xbins=W, cuda=args.cuda)


# Model
print('==> Building model..')
net = SSD300(num_classes=CLASSES, in_channels=CIN, height=H, width=W)

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

if args.visdom:
    viz = visdom.Visdom(port=args.port)
    vis_title = 'ConvRNN-SSD_' + str(H) + 'x' + str(W)
    vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
    iter_plot = viz.line(X=torch.zeros((1,)).cpu(),
                         Y=torch.zeros((1, 3)).cpu(),
                         opts=dict(
                             xlabel='Iteration',
                             ylabel='Loss',
                             title=vis_title,
                             legend=vis_legend)
                         )

    if isinstance(dataset, SquaresImages):
        image_plot = viz.images(np.random.randn(args.batchsize, 3, img_size[0], img_size[1]),
                                opts=dict(title='Eval images'))

# =============================================================================
#     if isinstance(dataset,SquaresVideos) or isinstance(dataset,H5SequenceDetection):
#         ncols = args.batchsize / nrows
#         video = np.random.rand(T, nrows*img_size[0], ncols*img_size[1], 3).astype(np.uint8)
#         video_plot = viz.video(tensor=video)
# =============================================================================


criterion = SSDLoss(num_classes=CLASSES)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)


# Training
def train(epoch, max_iter=500, log_every=10, save_every=50):
    print('\nEpoch: %d' % epoch)
    net.train()
    net.reset()
    train_loss = 0

    for batch_idx in range(max_iter):
        inputs, targets = dataset.next()

        # if args.reset_each_batch:
        #    net.reset()

        if args.cuda:
            inputs = inputs.cuda()
            targets = [y.cuda() for y in targets]

        optimizer.zero_grad()
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

        loc_loss, cls_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss = loc_loss + cls_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()

        if args.visdom:
            loc = loc_loss.data.item()
            conf = cls_loss.data.item()

            viz.line(X=torch.ones((1, 3)).cpu() * (batch_idx + epoch * max_iter),
                     Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / 20,
                     win=iter_plot,
                     update='append')

        if batch_idx % log_every == 0:
            print('\rtrain_loss: %.3f | avg_loss: %.3f [%d/%d]'
                  % (loss.data.item(), train_loss / (batch_idx + 1), batch_idx + 1, len(dataset)), ' ')

        if batch_idx % save_every == 0:
            # Save checkpoint (Here Train & Test is globally the same)
            global best_loss
            test_loss = loss.data.item()
            if test_loss < best_loss:
                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'loss': test_loss,
                    'epoch': epoch,
                }
                if not os.path.isdir(os.path.dirname(args.checkpoint)):
                    os.mkdir(os.path.dirname(args.checkpoint))
                torch.save(state, args.checkpoint)
                best_loss = test_loss


def _dig_out_time(x, n=32):
    nt, nanchors, c = x.size()
    t = int(nt / n)
    x = x.view(n, t, nanchors, c)
    return x


def test(epoch, max_iter=10):
    print('\nEpoch (test): %d' % epoch)
    net.eval()
    net.reset()

    if isinstance(dataset, SquaresImages):
        inputs, _ = dataset.next()
        images = inputs.cpu().data.numpy()

        if args.cuda:
            inputs = inputs.cuda()

        loc_preds, cls_preds = net(inputs)
        vis_show = np.zeros((args.batchsize, 3, img_size[0], img_size[1]), dtype=np.uint8)
        for i in range(loc_preds.size(0)):
            img = make_single_channel_display(images[i, 0])
            boxes, labels, scores = box_coder.decode(loc_preds[i].data, F.softmax(cls_preds[i], dim=1).data)

            if boxes is not None:
                bboxes = boxarray_to_boxes(boxes, labels, dataset.labelmap)
                img = draw_bboxes(img, bboxes)
            vis_show[i] = np.moveaxis(img, 2, 0)

        if args.visdom:
            viz.images(vis_show, win=image_plot)

    elif isinstance(dataset, SquaresVideos) or isinstance(dataset, H5SequenceDetection):

        periods = max_iter

        batchsize = dataset.batchsize
        time = dataset.time

        ncols = batchsize / nrows

        grid = np.zeros((time * periods, nrows, ncols, img_size[0], img_size[1], 3), dtype=np.uint8)
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
                    if CIN == 3:
                        img = np.moveaxis(images[i, :, t], 0, 2)
                        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                        show = np.zeros((H, W, 3), dtype=np.float32)
                        show[...] = img
                        img = show
                    elif CIN == 2:
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
        video = video.reshape(time * periods, nrows * img_size[0], ncols * img_size[1],
                              3)  # (T,Rows,H,Cols,W,3) -> (T,Rows*H,Cols*W,3)

        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.cv.FOURCC(*'XVID')
        videofile = './checkpoints/tests/' + str(epoch) + '.avi'
        print('writing to: ', videofile)
        out = cv2.VideoWriter(videofile, fourcc, 30.0, (ncols * img_size[1], nrows * img_size[0]))
        for t in range(time * periods):
            out.write(video[t])
        out.release()

        # viz.video(tensor=video, win=video_plot)
        net.extractor.return_all = False


def save_ckpt(epoch, name='ckpt'):
    state = {
        'net': net.state_dict(),
        'loss': best_loss,
        'epoch': epoch,
    }
    ckpt_file = os.path.dirname(args.checkpoint) + '/ckpt_' + str(epoch) + '.pth'
    if not os.path.isdir(os.path.dirname(args.checkpoint)):
        os.mkdir(os.path.dirname(args.checkpoint))
    torch.save(state, ckpt_file)


def main():
    for epoch in range(start_epoch, start_epoch + 200):
        box_coder.iou_threshold = min(0.5, 0.3+epoch*0.05)
        train(epoch, args.max_iter, 10, 50)
        test(epoch, args.test_iter)
        # scheduler.step()
        save_ckpt(epoch)

        # box_coder.iou_threshold = max(box_coder.iou_threshold*1.1,0.5)
        # print('box_coder current threshold: ', box_coder.iou_threshold)


if __name__ == '__main__':
    main()