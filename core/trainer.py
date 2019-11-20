from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import collections
import torch
import numpy as np
from tensorboardX import SummaryWriter
from core.utils import vis, tbx, plot, image
from core.eval import mean_ap
import cv2
from tqdm import tqdm

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

    def train(self, epoch, dataloader, args):
        print('\nEpoch: %d (train)' % epoch)
        self.net.train()
        self.net.reset()
        train_loss = 0
        dataloader.dataset.max_consecutive_batches = 4 #(2 ** epoch)
        dataloader.dataset.build()

        loss_hist = collections.deque(maxlen=500)
        start = 0
        runtime_stats = {'dataloader': 0, 'network': 0}
        for batch_idx, data in enumerate(dataloader):
            if batch_idx > 0:
                runtime_stats['dataloader'] += time.time() - start
            inputs, targets, reset = data['data'], data['boxes'], data['resets']
            if args.cuda:
                inputs = inputs.cuda()

            if reset:
                self.net.reset()

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
                                       batch_idx + epoch * len(dataloader) )

            if batch_idx % args.log_every == 0:
                print('\rtrain_loss: %.3f | avg_loss: %.3f [%d/%d] | @data: %.3f | @net: %.3f'
                      % (loss, train_loss / (batch_idx + 1),
                         batch_idx + 1, len(dataloader),
                         runtime_stats['dataloader'] / (batch_idx + 1),
                         runtime_stats['network'] / (batch_idx + 1)
                         ), ' ')

            start = time.time()
            loss_hist.append(loss.item())
        return loss_hist

    def evaluate(self, epoch, dataloader, args):
        print('\nEpoch: %d (val)' % epoch)
        self.net.eval()
        self.net.reset()
        val_loss = 0
        #TODO: set this depending on curriculum argument
        dataloader.dataset.max_consecutive_batches = 4 #(2 ** epoch)
        dataloader.dataset.build()

        gts = [] #list of K array of size 5
        proposals = [] #list of K array of size 6
        start = 0
        runtime_stats = {'dataloader': 0, 'network': 0}
        for batch_idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            if batch_idx > 0:
                runtime_stats['dataloader'] += time.time() - start
            inputs, targets, reset = data['data'], data['boxes'], data['resets']
            if args.cuda:
                inputs = inputs.cuda()

            if reset:
                self.net.reset()

            with torch.no_grad():
                start = time.time()
                preds = self.net.get_boxes(inputs, score_thresh=0.1)
                runtime_stats['network'] += time.time() - start

            for t in range(len(targets)):
                for i in range(len(targets[t])):
                    gt_boxes = targets[t][i]
                    gt_boxes[..., 4] -= dataloader.dataset.label_offset
                    boxes, labels, scores = preds[t][i]

                    gts.append(gt_boxes.cpu().numpy())
                    if boxes is None:
                        pred = np.zeros((0, 6), dtype=np.float32)
                    else:
                        boxes, labels, scores = boxes.cpu(), labels.cpu()[:,None].float(), scores.cpu()[:,None]
                        pred = torch.cat([boxes, scores, labels], dim=1).numpy()

                    proposals.append(pred)

            start = time.time()

        det_results, gt_bboxes, gt_labels = mean_ap.convert(gts, proposals, self.net.num_classes)


        map_50, eval_results = mean_ap.eval_map(det_results,
                                                 gt_bboxes,
                                                 gt_labels,
                                                 gt_ignore=None,
                                                 scale_ranges=None,
                                                 iou_thr=0.5,
                                                 dataset=None,
                                                 print_summary=True)
        
        #compute actual mean_ap of coco
        mean_ap_levels = [map_50]
        for iou_threshold in np.linspace(0.55, 0.95, 9).tolist():
            mean_ap_level, _ = mean_ap.eval_map(det_results,
                                                gt_bboxes,
                                                gt_labels,
                                                gt_ignore=None,
                                                scale_ranges=None,
                                                iou_thr=iou_threshold,
                                                dataset=None,
                                                print_summary=False)

            print('iou_threshold: ', iou_threshold, mean_ap_level)
            mean_ap_levels.append(mean_ap_level)
        mean_ap_coco = sum(mean_ap_levels)/len(mean_ap_levels)

        self.writer.add_scalar('mean_ap_coco', mean_ap_coco, epoch)
        self.writer.add_scalar('mean_ap_50', map_50, epoch)
        xs = [item['recall'] for item in eval_results]
        ys = [item['precision'] for item in eval_results]
        fig = plot.xy_curves(xs, ys)
        self.writer.add_figure('pr_curves', fig, epoch)

        aps = np.array([item['ap'] for item in eval_results])
        fig2 = plot.bar(aps)
        self.writer.add_figure('aps', fig2, epoch)
        return map_50


    def test(self, epoch, dataloader, args):
        print('\nEpoch: %d (test)' % epoch)
        self.net.eval()
        self.net.reset()

        dataloader.dataset.reset()

        labelmap = dataloader.dataset.labelmap
        batchsize = args.batchsize
        time = args.time
        nrows = 2 ** ((batchsize.bit_length() - 1) // 2)
        ncols = batchsize // nrows
        
        if args.is_video_dataset:
            grid = np.zeros((args.test_iter * time, nrows, ncols, args.height, args.width, 3), dtype=np.uint8)
        
        for period, data in tqdm(enumerate(dataloader), total=args.test_iter):
            inputs, targets, reset = data['data'], data['boxes'], data['resets']
            images = inputs.cpu().clone().data.numpy()

            if args.cuda:
                inputs = inputs.cuda()

            if reset:
                self.net.reset()

            with torch.no_grad():
                targets = self.net.get_boxes(inputs)

            time, batchsize, _, height, width = images.shape

            if args.is_video_dataset:
                vis.draw_txn_boxes_on_grid(images, targets, grid[period * time:], self.make_image, labelmap)
            else:
                grid2 = np.zeros((time, nrows, ncols, height, width, 3), dtype=np.uint8)
                vis.draw_txn_boxes_on_grid(images, targets, grid2, self.make_image, labelmap)
                grid2 = grid2.swapaxes(2, 3).reshape(nrows * height, ncols * width, 3)
                image_name = self.logdir + '/images/test_batch#'+str(period)+'.jpg'
                tbx.prepare_ckpt_dir(image_name)
                cv2.imwrite(image_name, grid2[...,::-1])

            if period >= (args.test_iter-1):
                break

        if args.is_video_dataset:
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
