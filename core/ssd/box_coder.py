from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
from core.utils.box import box_soft_nms, box_nms, change_box_order, assign_priors
from core.utils import opts


class SSDBoxCoder(torch.nn.Module):
    def __init__(self, ssd_model, fg_iou_threshold=0.6, bg_iou_threshold=0.4, soft_nms=False):
        super(SSDBoxCoder, self).__init__()

        self.steps = ssd_model.steps
        self.box_sizes = ssd_model.box_sizes
        self.aspect_ratios = ssd_model.aspect_ratios
        self.fm_sizes = ssd_model.fm_sizes
        self.height = ssd_model.height
        self.width = ssd_model.width
        self.fm_len = []
        self.register_buffer('default_boxes', self._get_default_boxes())
        self.register_buffer('default_boxes_xyxy', change_box_order(self.default_boxes, 'xywh2xyxy'))
        self.fg_iou_threshold = fg_iou_threshold
        self.bg_iou_threshold = bg_iou_threshold
        self.use_cuda = False
        self.variances = (0.1, 0.1)
        self.nms = box_soft_nms if soft_nms else box_nms

    def reset(self, ssd_model):
        self.steps = ssd_model.steps
        self.box_sizes = ssd_model.box_sizes
        self.aspect_ratios = ssd_model.aspect_ratios
        self.fm_sizes = ssd_model.fm_sizes
        self.height = ssd_model.height
        self.width = ssd_model.width
        self.fm_len = []
        self.register_buffer('default_boxes', self._get_default_boxes())
        self.register_buffer('default_boxes_xyxy', change_box_order(self.default_boxes, 'xywh2xyxy'))

    def __greet__(self):
        print('steps', self.steps)
        print('box_sizes', self.box_sizes)
        print('aspect_ratios', self.aspect_ratios)
        print('fm_sizes', self.fm_sizes)
        print('img size', self.height, self.width)
        print('default_boxes', self.default_boxes)
        
    def reset(self, ssd_model):
        self.steps = ssd_model.steps
        self.box_sizes = ssd_model.box_sizes
        self.aspect_ratios = ssd_model.aspect_ratios
        self.fm_sizes = ssd_model.fm_sizes
        self.height = ssd_model.height
        self.width = ssd_model.width
        self.default_boxes = self._get_default_boxes()
        self.default_boxes_xyxy =  change_box_order(self.default_boxes, 'xywh2xyxy')

    def _get_default_boxes(self):
        boxes = []
        nanchors = 2 * len(self.aspect_ratios) + 2
        for i, fm_size in enumerate(self.fm_sizes):
            f_y, f_x = fm_size
            s_y, s_x = self.steps[i]
            print('sy, sx: ', s_y, s_x)

            self.fm_len.append(f_y * f_x * nanchors)
            s = self.box_sizes[i]
            for h in range(f_y):
                for w in range(f_x):
                    cx = (w + 0.5) * s_x
                    cy = (h + 0.5) * s_y

                    boxes.append((cx, cy, s, s))

                    s = math.sqrt(self.box_sizes[i] * self.box_sizes[i+1])
                    boxes.append((cx, cy, s, s))

                    s = self.box_sizes[i]
                    for ar in self.aspect_ratios:
                        boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                        boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))

        boxes = torch.Tensor(boxes)
        return boxes

    def encode(self, gt_boxes, labels):
        boxes, cls_targets = assign_priors(gt_boxes, labels + 1, self.default_boxes_xyxy,
                                            self.fg_iou_threshold, self.bg_iou_threshold)
        boxes = change_box_order(boxes, 'xyxy2xywh')
        default_boxes = self.default_boxes
        loc_xy = (boxes[:, :2] - default_boxes[:, :2]) / default_boxes[:, 2:] / self.variances[0]
        loc_wh = torch.log(boxes[:, 2:] / default_boxes[:, 2:]) / self.variances[1]
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        return loc_targets, cls_targets

    def decode_loc(self, loc_preds):
        if loc_preds.dim() > self.default_boxes.dim():
            default_boxes = self.default_boxes[None, ...]
        else:
            default_boxes = self.default_boxes

        xy = loc_preds[..., :2] * self.variances[0] * default_boxes[..., 2:] + default_boxes[..., :2]
        wh = torch.exp(loc_preds[..., 2:] * self.variances[1]) * default_boxes[..., 2:]
        box_preds = torch.cat([xy - wh / 2, xy + wh / 2], -1)
        return box_preds

    def decode(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        xy = loc_preds[:,:2] * self.variances[0] * self.default_boxes[:,2:] + self.default_boxes[:,:2]
        wh = torch.exp(loc_preds[:,2:] * self.variances[1]) * self.default_boxes[:,2:]

        box_preds = torch.cat([xy-wh/2, xy+wh/2], 1)

        boxes = []
        labels = []
        scores = []
        num_classes = cls_preds.size(1)


        for i in range(num_classes-1):
            score = cls_preds[:,i+1]  # class i corresponds to (i+1) column
            mask = score > score_thresh
            if not mask.any():
                continue

            box = box_preds[mask]
            score = score[mask]

            if nms_thresh == 1.0:
                boxes.append(box)
                labels.append(torch.LongTensor(len(box)).fill_(i))
                scores.append(score)
            else:
                keep = self.nms(box, score, nms_thresh)
                boxes.append(box[keep])
                labels.append(torch.LongTensor(len(box[keep])).fill_(i))
                scores.append(score[keep])


        if len(boxes) > 0:
            boxes = torch.cat(boxes, 0)
            labels = torch.cat(labels, 0)
            scores = torch.cat(scores, 0)
            return boxes, labels, scores
        else:
            return None, None, None

    def encode_txn_boxes(self, targets):
        loc_targets, cls_targets = [], []

        device = self.default_boxes.device

        for t in range(len(targets)):
            for i in range(len(targets[t])):
                boxes, labels = targets[t][i][:, :-1], targets[t][i][:, -1]
                boxes, labels = boxes.to(device), labels.to(device)

                loc_t, cls_t = self.encode(boxes, labels)
                loc_targets.append(loc_t.unsqueeze(0))
                cls_targets.append(cls_t.unsqueeze(0).long())

        loc_targets = torch.cat(loc_targets, dim=0)  # (N,#anchors,4)
        cls_targets = torch.cat(cls_targets, dim=0)  # (N,#anchors,C)

        return loc_targets, cls_targets

    def decode_txn_boxes(self, loc_preds, cls_preds, batchsize):
        loc_preds = opts.batch_to_time(loc_preds, batchsize)
        cls_preds = opts.batch_to_time(cls_preds, batchsize)
        targets = []
        for t in range(loc_preds.size(0)):
            targets_t = []
            for i in range(loc_preds.size(1)):
                boxes, labels, scores = self.decode(loc_preds[t, i].data,
                                                                cls_preds[t, i].data,
                                                                nms_thresh=0.6)
                targets_t.append((boxes, labels, scores))
            targets.append(targets_t)
        return targets
