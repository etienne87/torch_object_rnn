from __future__ import print_function
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
import utils


class SSDTrainer(object):
    """
    class wrapping training/ validation/ testing
    """

    def __init__(self, logdir, net, box_coder, criterion, optimizer, all_timesteps=False):
        self.net = net
        self.box_coder = box_coder
        self.criterion = criterion
        self.optimizer = optimizer
        self.all_timesteps = all_timesteps
        self.make_image = utils.general_frame_display
        self.writer = SummaryWriter(logdir)

    def __del__(self):
        self.writer.close()

    def train(self, epoch, dataset, args):
        print('\nEpoch: %d (train)' % epoch)
        self.net.train()
        self.net.reset()
        train_loss = 0
        self.net.extractor.return_all = self.all_timesteps

        dataset.reset()
        proba_reset = 1 * (0.9)**epoch

        start = 0
        runtime_stats = {'dataloader': 0, 'network': 0}
        for batch_idx, data in enumerate(dataset):
            if batch_idx > 0:
                runtime_stats['dataloader'] += time.time() - start
            inputs, targets = data
            if args.cuda:
                inputs = inputs.cuda()

            if np.random.rand() < proba_reset:
                dataset.reset()
                self.net.reset()

            start = time.time()
            self.optimizer.zero_grad()
            loc_preds, cls_preds = self.net(inputs)
            loc_targets, cls_targets = utils.encode_boxes(targets, self.box_coder, args.cuda)

            loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss = loc_loss + cls_loss
            loss.backward()
            self.optimizer.step()

            runtime_stats['network'] += time.time() - start

            train_loss += loss.item()

            self.writer.add_scalar('train_loss', loss.data.item(), batch_idx + epoch * len(dataset))

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

        start = 0
        runtime_stats = {'dataloader': 0, 'network': 0}
        for batch_idx, data in enumerate(dataset):
            if batch_idx > 0:
                runtime_stats['dataloader'] += time.time() - start
            inputs, targets = data
            if args.cuda:
                inputs = inputs.cuda()

            start = time.time()
            loc_preds, cls_preds = self.net(inputs)
            loc_targets, cls_targets = utils.encode_boxes(targets, self.box_coder, args.cuda)
            loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss = loc_loss + cls_loss
            val_loss += loss.item()
            runtime_stats['network'] += time.time() - start

            if batch_idx % args.log_every == 0:
                print('\rval_loss: %.3f | avg_loss: %.3f [%d/%d] | @data: %.3f | @net: %.3f'
                      % (loss.data.item(), val_loss / (batch_idx + 1), batch_idx + 1, len(dataset),
                         runtime_stats['dataloader'] / (batch_idx + 1),
                         runtime_stats['network'] / (batch_idx + 1)
                         ), ' ')

            start = time.time()

        self.writer.add_scalar('val_loss', loss.data.item(), epoch)

    def test(self, epoch, dataset, nrows, args):
        print('\nEpoch: %d (test)' % epoch)
        self.net.eval()
        self.net.reset()
        self.net.extractor.return_all = True

        dataset.reset()

        periods = args.test_iter
        batchsize = dataset.batchsize
        time = dataset.time

        ncols = batchsize / nrows
        grid = np.zeros((periods * time, nrows, ncols, dataset.height, dataset.width, 3), dtype=np.uint8)

        for period in range(periods):
            inputs, targets = dataset.next()

            if args.cuda:
                inputs = inputs.cuda()

            loc_preds, cls_preds = self.net(inputs)

            images = inputs.cpu().data.numpy()
            loc_preds = utils.dig_out_time(loc_preds, dataset.batchsize)
            cls_preds = utils.dig_out_time(cls_preds, dataset.batchsize)
            for t in range(loc_preds.size(0)):
                for i in range(loc_preds.size(1)):
                    y = i / ncols
                    x = i % ncols
                    img = self.make_image(images[t, i])
                    # assert img.shape == grid[0, 0].shape
                    boxes, labels, scores = self.box_coder.decode(loc_preds[t, i].data,
                                                                  cls_preds[t, i].data,
                                                                  nms_thresh=0.4)
                    if boxes is not None:
                        bboxes = utils.boxarray_to_boxes(boxes, labels, dataset.labelmap)
                        img = utils.draw_bboxes(img, bboxes)

                    grid[t + period * time, y, x] = img

        grid = grid.swapaxes(2, 3).reshape(periods * time, nrows * dataset.height, ncols * dataset.width, 3)
        utils.add_video(self.writer, 'test', grid, global_step=epoch, fps=30)
        self.net.extractor.return_all = False

    def save_ckpt(self, epoch, args, name='checkpoint#'):
        state = {
            'net': self.net.state_dict(),
            'epoch': epoch,
        }
        ckpt_file = os.path.dirname(args.checkpoint) + '/' + name + str(epoch) + '.pth'
        utils.prepare_ckpt_dir(ckpt_file)
        torch.save(state, ckpt_file)
