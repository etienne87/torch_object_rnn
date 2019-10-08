from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np
from tensorboardX import SummaryWriter
from core.utils import vis, tbx, plot
from core.eval import mean_ap






class DetTrainer(object):
    """
    class wrapping training/ validation/ testing
    """

    def __init__(self, logdir, net, optimizer):
        self.net = net
        self.optimizer = optimizer
        self.make_image = vis.general_frame_display
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)

    def __del__(self):
        self.writer.close()

    def _progressive_resize(self, epoch, net, train_dataset, test_dataset):
        if epoch < 4:
            size = 128
        elif epoch < 11:
            size = 256
        else:
            size = 512

        train_dataset.resize(size, size)
        test_dataset.resize(size, size)
        net.resize(size, size)
        net.box_coder.cuda()
        print('size: ', size)

    def train(self, epoch, dataloader, args):
        print('\nEpoch: %d (train)' % epoch)
        self.net.train()
        self.net.reset()
        train_loss = 0
        proba_reset = 1 * (0.9)**epoch

        start = 0
        runtime_stats = {'dataloader': 0, 'network': 0}
        for batch_idx, data in enumerate(dataloader):
            if batch_idx > 0:
                runtime_stats['dataloader'] += time.time() - start
            inputs, targets = data
            if args.cuda:
                inputs = inputs.cuda()

            if np.random.rand() < proba_reset:
                if hasattr(dataloader.dataset, "reset"):
                    dataloader.dataset.reset()
                self.net.reset()

            # deal with permanent reset with a mask
            # self.net.reset()

            start = time.time()
            self.optimizer.zero_grad()
            loss_dict = self.net.compute_loss(inputs, targets)

            loss = sum([value for key, value in loss_dict.items()])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.1)
            self.optimizer.step()

            runtime_stats['network'] += time.time() - start

            train_loss += loss.item()

            for key, value in loss_dict.items():
                self.writer.add_scalar('train_'+key, value.data.item(),
                                       batch_idx + epoch * len(dataloader.dataset) // args.batchsize )

            if batch_idx % args.log_every == 0:
                print('\rtrain_loss: %.3f | avg_loss: %.3f [%d/%d] | @data: %.3f | @net: %.3f'
                      % (loss.data.item(), train_loss / (batch_idx + 1),
                         batch_idx + 1, len(dataloader.dataset) // args.batchsize,
                         runtime_stats['dataloader'] / (batch_idx + 1),
                         runtime_stats['network'] / (batch_idx + 1)
                         ), ' ')

            start = time.time()

    def evaluate(self, epoch, dataloader, args):
        print('\nEpoch: %d (val)' % epoch)
        self.net.eval()
        self.net.reset()
        val_loss = 0

        gts = [] #list of K array of size 5
        proposals = [] #list of K array of size 6
        start = 0
        runtime_stats = {'dataloader': 0, 'network': 0}
        for batch_idx, data in enumerate(dataloader):
            if batch_idx > 0:
                runtime_stats['dataloader'] += time.time() - start
            inputs, targets = data
            if args.cuda:
                inputs = inputs.cuda()

            if batch_idx%10 == 0:
                if hasattr(dataloader.dataset, "reset"):
                    dataloader.dataset.reset()
                self.net.reset()

            # self.net.reset()

            with torch.no_grad():
                start = time.time()
                preds = self.net.get_boxes(inputs, score_thresh=0.1)
                runtime_stats['network'] += time.time() - start

            for t in range(len(targets)):
                for i in range(len(targets[t])):
                    gt_boxes = targets[t][i]
                    boxes, labels, scores = preds[t][i]

                    gts.append(gt_boxes.cpu().numpy())
                    if boxes is None:
                        pred = np.zeros((0, 6), dtype=np.float32)
                    else:
                        boxes, labels, scores = boxes.cpu(), labels.cpu()[:,None].float(), scores.cpu()[:,None]
                        pred = torch.cat([boxes, scores, labels], dim=1).numpy()

                    proposals.append(pred)

            start = time.time()

        det_results, gt_bboxes, gt_labels = mean_ap.convert(gts, proposals, self.net.num_classes-1)

        map, eval_results = mean_ap.eval_map(det_results,
                                                 gt_bboxes,
                                                 gt_labels,
                                                 gt_ignore=None,
                                                 scale_ranges=None,
                                                 iou_thr=0.5,
                                                 dataset=None,
                                                 print_summary=True)

        self.writer.add_scalar('mean_ap', map, epoch)
        xs = [item['recall'] for item in eval_results]
        ys = [item['precision'] for item in eval_results]
        fig = plot.xy_curves(xs, ys)
        self.writer.add_figure('pr_curves', fig, epoch)

        aps = np.array([item['ap'] for item in eval_results])
        fig2 = plot.bar(aps)
        self.writer.add_figure('aps', fig2, epoch)
        return map


    def test(self, epoch, dataloader, args):
        print('\nEpoch: %d (test)' % epoch)
        self.net.eval()
        self.net.reset()

        if hasattr(dataloader.dataset, "reset"):
            dataloader.dataset.reset()

        batchsize = args.batchsize
        time = args.time
        labelmap = dataloader.dataset.labelmap
        nrows = 2 ** ((batchsize.bit_length() - 1) // 2)
        ncols = batchsize // nrows
        grid = np.zeros((args.test_iter * time, nrows, ncols, args.height, args.width, 3), dtype=np.uint8)

        for period, (inputs, targets) in enumerate(dataloader):
            images = inputs.clone().data.numpy()

            if args.cuda:
                inputs = inputs.cuda()

            self.net.reset()
            with torch.no_grad():
                targets = self.net.get_boxes(inputs)

            vis.draw_txn_boxes_on_images(images, targets, grid, self.make_image,
                                         period, time, ncols, labelmap)

            if period >= (args.test_iter-1):
                break

        grid = grid.swapaxes(2, 3).reshape(args.test_iter * time, nrows * args.height, ncols * args.width, 3)


        if args.save_video:
            video_name =  self.logdir + '/videos/' + 'video#' + str(epoch) + '.avi'
            tbx.prepare_ckpt_dir(video_name)
            vis.write_video_opencv(video_name, grid)
        else:
            tbx.add_video(self.writer, 'test', grid[...,::-1], global_step=epoch, fps=30)


    def save_ckpt(self, epoch, name='checkpoint#'):
        state = {
            'net': self.net.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict()
        }
        ckpt_file = self.logdir + '/checkpoints/' + name + '.pth'
        tbx.prepare_ckpt_dir(ckpt_file)
        torch.save(state, ckpt_file)
