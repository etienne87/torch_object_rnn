from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
from core.utils.box import box_soft_nms, box_nms, np_box_nms, change_box_order, assign_priors, box_iou
from core.utils import opts



def get_box_params_variable_size(sources, h, w):
    image_size = float(min(h, w))
    steps = []
    box_sizes = []
    fm_sizes = []
    s_min, s_max = 0.1, 0.9
    m = float(len(sources))
    for k, src in enumerate(sources):
        # featuremap size
        fm_sizes.append((src.size(2), src.size(3)))
        # step is ratio image_size / featuremap_size
        step_y, step_x = math.floor(float(h) / src.size(2)), math.floor(float(w) / src.size(3))
        steps.append((step_y, step_x))
        # compute scale
        s_k = s_min + (s_max - s_min) * k / m
        # box_size is scale * image_size
        box_sizes.append(math.floor(s_k * image_size))
        print("box size: ", box_sizes[-1])
    s_k = s_min + (s_max - s_min)
    box_sizes.append(s_k * image_size)
    return fm_sizes, steps, box_sizes


def get_box_params_fixed_size(sources, h, w):
    steps = []
    box_sizes = []
    fm_sizes = []
    for k, src in enumerate(sources):
        # featuremap size
        fm_sizes.append((src.size(2), src.size(3)))
        # step is ratio image_size / featuremap_size
        step_y, step_x = math.floor(float(h) / src.size(2)), math.floor(float(w) / src.size(3))
        steps.append((step_y, step_x))
        # compute scale
        box_sizes.append(24 * 2**k)
        print("box size: ", box_sizes[-1])
    box_sizes.append(24 * 2**k)
    return fm_sizes, steps, box_sizes


class SSDBoxCoder(torch.nn.Module):
    def __init__(self, ssd_model, fg_iou_threshold=0.6, bg_iou_threshold=0.4, soft_nms=False):
        super(SSDBoxCoder, self).__init__()

        self.steps = ssd_model.steps
        self.box_sizes = ssd_model.box_sizes
        self.aspect_ratios = ssd_model.aspect_ratios
        self.scales = ssd_model.scales
        self.fm_sizes = ssd_model.fm_sizes
        self.height = ssd_model.height
        self.width = ssd_model.width
        self.fm_len = []
        self.register_buffer('default_boxes', self._get_default_boxes_v2())
        self.register_buffer('default_boxes_xyxy', change_box_order(self.default_boxes, 'xywh2xyxy'))
        self.fg_iou_threshold = fg_iou_threshold
        self.bg_iou_threshold = bg_iou_threshold
        self.use_cuda = False
        self.variances = (0.1, 0.2)
        self.nms = box_soft_nms if soft_nms else box_nms
        self.encode = self.encode_fast

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

    def _get_default_boxes_v2(self):
        boxes = []
        num_anchors = len(self.aspect_ratios) * len(self.scales)

        for i, fm_size in enumerate(self.fm_sizes):
            f_y, f_x = fm_size
            s_y, s_x = self.steps[i]
            print('sy, sx: ', s_y, s_x)

            self.fm_len.append(f_y * f_x * num_anchors)
            base_size = self.box_sizes[i]
            for h in range(f_y):
                for w in range(f_x):
                    cx = (w + 0.5) * s_x
                    cy = (h + 0.5) * s_y

                    for aspect_ratio in self.aspect_ratios:
                        for scale in self.scales:
                            w2 = base_size * scale * math.sqrt(aspect_ratio)
                            h2 = base_size * scale / math.sqrt(aspect_ratio)
                            boxes.append((cx, cy, w2, h2))

        boxes = torch.Tensor(boxes)
        return boxes

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

    def encode_alt(self, boxes, gt_labels):
        labels = gt_labels + 1
        default_boxes = self.default_boxes_xyxy
        ious = box_iou(default_boxes, boxes)  # [#anchors, #obj]
        index = torch.zeros(len(default_boxes), dtype=torch.int64, device=boxes.device).fill_(-1)

        # We match every ground truth with higher than iou_threshold iou with an anchor
        max_iou_anchors, arg_max_iou_anchors = torch.max(ious, dim=1)
        mask_greater_than_threshold = max_iou_anchors >= self.fg_iou_threshold
        mask_ambiguous = (max_iou_anchors >= self.bg_iou_threshold) * (max_iou_anchors < self.fg_iou_threshold)
        index[mask_greater_than_threshold] = arg_max_iou_anchors[mask_greater_than_threshold]
        index[mask_ambiguous] = -1

        # We match every ground truth to an anchor the maximum iou iteratively (avoid two gt matched with same anchor)
        for index_gt in range(len(boxes)):
            arg_max_iou_gt = torch.argmax(ious[:, index_gt])
            if ious[arg_max_iou_gt, index_gt] >= 0.2:
                index[arg_max_iou_gt] = index_gt
                ious[arg_max_iou_gt, :] = 0

        boxes = boxes[index.clamp(min=0)]  # negative index not supported
        boxes = change_box_order(boxes, 'xyxy2xywh')
        default_boxes = self.default_boxes  # change_box_order(default_boxes, 'xyxy2xywh')

        variances = (0.1, 0.2)
        loc_xy = (boxes[:, :2] - default_boxes[:, :2]) / default_boxes[:, 2:] / variances[0]
        loc_wh = torch.log(boxes[:, 2:] / default_boxes[:, 2:]) / variances[1]
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        cls_targets = labels[index.clamp(min=0)]
        cls_targets[index < 0] = 0
        return loc_targets, cls_targets

    def encode_fast(self, gt_boxes, labels):
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

    def multiclass_nms(self, box_preds, cls_preds, score_thresh=0.5, nms_thresh=0.45):
        boxes = []
        labels = []
        scores = []
        num_classes = cls_preds.shape[1]

        for i in range(num_classes-1):
            score = cls_preds[:,i+1]  # class i corresponds to (i+1) column
            mask = score > score_thresh
            if not mask.any():
                continue

            box = box_preds[mask]
            score = score[mask]

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

    def encode_boxes(self, targets):
        loc_targets, cls_targets = [], []

        device = self.default_boxes.device

        for t in range(len(targets)):
            if len(targets[t]) == 0:
                targets[t] = torch.ones((1, 5), dtype=torch.float32)*-1

            boxes, labels = targets[t][:, :4], targets[t][:, -1]
            boxes, labels = boxes.to(device), labels.to(device)

            loc_t, cls_t = self.encode(boxes, labels)
            loc_targets.append(loc_t.unsqueeze(0))
            cls_targets.append(cls_t.unsqueeze(0).long())

        loc_targets = torch.cat(loc_targets, dim=0)  # (N,#anchors,4)
        cls_targets = torch.cat(cls_targets, dim=0)  # (N,#anchors,C)
        return loc_targets, cls_targets

    def encode_txn_boxes(self, targets):
        # import time
        # start = time.time()
        loc_targets, cls_targets = [], []

        device = self.default_boxes.device

        for t in range(len(targets)):
            for i in range(len(targets[t])):
                if len(targets[t][i]) == 0:
                    targets[t][i] = torch.ones((1, 5), dtype=torch.float32)*-1

                boxes, labels = targets[t][i][:, :4], targets[t][i][:, -1]
                boxes, labels = boxes.to(device), labels.to(device)

                loc_t, cls_t = self.encode(boxes, labels)
                loc_targets.append(loc_t.unsqueeze(0))
                cls_targets.append(cls_t.unsqueeze(0).long())

        loc_targets = torch.cat(loc_targets, dim=0)  # (N,#anchors,4)
        cls_targets = torch.cat(cls_targets, dim=0)  # (N,#anchors,C)

        # print((time.time()-start), ' s for encoding')

        return loc_targets, cls_targets

    def decode_txn_boxes(self, loc_preds, cls_preds, batchsize, score_thresh):
        box_preds = self.decode_loc(loc_preds)

        box_preds = opts.batch_to_time(box_preds, batchsize).data
        cls_preds = opts.batch_to_time(cls_preds, batchsize).data

        tbins, batchsize = box_preds.shape[:2]


        box_preds = box_preds.cpu()
        cls_preds = cls_preds.cpu()

        if self.nms == np_box_nms:
            box_preds = box_preds.cpu().numpy()
            cls_preds = cls_preds.cpu().numpy()

        targets = []
        for t in range(tbins):
            targets_t = []
            for i in range(batchsize):
                boxes, labels, scores = self.multiclass_nms(box_preds[t, i],
                                                                cls_preds[t, i],
                                                                score_thresh=score_thresh,
                                                                nms_thresh=0.6)
                targets_t.append((boxes, labels, scores))
            targets.append(targets_t)
        return targets
