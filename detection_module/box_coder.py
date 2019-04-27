'''Encode object boxes and labels.'''
import math
import torch
from box import box_iou, box_nms, change_box_order


class SSDBoxCoder:
    def __init__(self, ssd_model):
        self.steps = ssd_model.steps
        self.box_sizes = ssd_model.box_sizes
        self.aspect_ratios = ssd_model.aspect_ratios
        self.fm_sizes = ssd_model.fm_sizes
        self.height = ssd_model.height
        self.width = ssd_model.width
        self.default_boxes = self._get_default_boxes()
        self.default_boxes_xyxy =  change_box_order(self.default_boxes, 'xywh2xyxy')
        self.iou_threshold = 0.5
        self.use_cuda = False
        self.variances = (0.2, 0.2)

    def reset(self, ssd_model):
        self.steps = ssd_model.steps
        self.box_sizes = ssd_model.box_sizes
        self.aspect_ratios = ssd_model.aspect_ratios
        self.fm_sizes = ssd_model.fm_sizes
        self.height = ssd_model.height
        self.width = ssd_model.width
        self.default_boxes = self._get_default_boxes()
        self.default_boxes_xyxy =  change_box_order(self.default_boxes, 'xywh2xyxy')

    def cuda(self):
          self.default_boxes = self.default_boxes.cuda()
          self.default_boxes_xyxy = self.default_boxes_xyxy.cuda()
          self.use_cuda = True

    def _get_default_boxes(self):
        boxes = []
        for i, fm_size in enumerate(self.fm_sizes):
            f_y, f_x = fm_size
            s_y, s_x = self.steps[i]
            print('sy, sx: ', s_y, s_x)
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

        return torch.Tensor(boxes)  # xywh

    def encode(self, boxes, labels):
        '''Encode target bounding boxes and class labels.

        SSD coding rules:
          tx = (x - anchor_x) / (variance[0]*anchor_w)
          ty = (y - anchor_y) / (variance[0]*anchor_h)
          tw = log(w / anchor_w) / variance[1]
          th = log(h / anchor_h) / variance[1]

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].

        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/multibox_coder.py
        '''
        def argmax(x):
            v, i = x.max(0)
            j = v.max(0)[1].item() #was (0)[1][0] which causes Pytorch Warning for Scalars
            return (i[j], j)

        default_boxes = self.default_boxes_xyxy

        ious = box_iou(default_boxes, boxes)  # [#anchors, #obj]

        if self.use_cuda:
            index = torch.cuda.LongTensor(len(default_boxes)).fill_(-1)
        else:
            index = torch.LongTensor(len(default_boxes)).fill_(-1)


        masked_ious = ious.clone()
        while True:
            i, j = argmax(masked_ious)
            if masked_ious[i,j] < 1e-6:
                break
            index[i] = j
            masked_ious[i,:] = 0
            masked_ious[:,j] = 0

        mask = (index<0) & (ious.max(1)[0]>=self.iou_threshold)
        if mask.any():
            # https: // github.com / kuangliu / torchcv / issues / 23
            index[mask] = ious[mask].max(1)[1]
            # index[mask] = ious[mask.nonzero().squeeze()].max(1)[1]

        boxes = boxes[index.clamp(min=0)]  # negative index not supported
        boxes = change_box_order(boxes, 'xyxy2xywh')
        default_boxes = self.default_boxes # change_box_order(default_boxes, 'xyxy2xywh')

        loc_xy = (boxes[:,:2]-default_boxes[:,:2]) / default_boxes[:,2:] / self.variances[0]
        loc_wh = torch.log(boxes[:,2:]/default_boxes[:,2:]) / self.variances[1]
        loc_targets = torch.cat([loc_xy,loc_wh], 1)
        cls_targets = 1 + labels[index.clamp(min=0)]
        cls_targets[index<0] = 0
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        '''Decode predicted loc/cls back to real box locations and class labels.

        Args:
          loc_preds: (tensor) predicted loc, sized [8732,4].
          cls_preds: (tensor) predicted conf, sized [8732,21].
          score_thresh: (float) threshold for object confidence score.
          nms_thresh: (float) threshold for box nms.

        Returns:
          boxes: (tensor) bbox locations, sized [#obj,4].
          labels: (tensor) class labels, sized [#obj,].
        '''
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

            #https://github.com/kuangliu/torchcv/issues/36
            #box = box_preds[mask.nonzero().squeeze()]
            box = box_preds[mask]
            score = score[mask]


            if nms_thresh == 1.0:
                boxes.append(box)
                labels.append(torch.LongTensor(len(box)).fill_(i))
                scores.append(score)
            else:
                keep = box_nms(box, score, nms_thresh)
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