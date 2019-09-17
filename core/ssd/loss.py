from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

def reduce(loss, mode='none'):
    if mode == 'mean':
        loss = loss.mean()
    elif mode == 'sum':
        loss = loss.sum()
    return loss


class SSDLoss(nn.Module):
    def __init__(self, num_classes, mode='focal', use_sigmoid=False):
        super(SSDLoss, self).__init__()
        self.num_classes = num_classes
        self.mode = mode
        self.alpha = torch.nn.Parameter(torch.ones(num_classes))
        self.focal_loss = self._sigmoid_focal_loss if use_sigmoid else self._softmax_focal_loss

    def _hard_negative_mining(self, cls_loss, pos):
        '''Return negative indices that is 3x the number as positive indices.

        Args:
          cls_loss: (tensor) cross entroy loss between cls_preds and cls_targets, sized [N,#anchors].
          pos: (tensor) positive class mask, sized [N,#anchors].

        Return:
          (tensor) negative indices, sized [N,#anchors].
        '''
        cls_loss = cls_loss * (pos.float() - 1)

        _, idx = cls_loss.sort(1)  # sort by negative losses
        _, rank = idx.sort(1)      # [N,#anchors]

        num_neg = 3*pos.sum(1)  # [N,]
        neg = rank < num_neg[:,None]   # [N,#anchors]
        return neg

    def _softmax_focal_loss(self, x, y, reduction='none'):
        '''Softmax Focal loss.

        Args:
          x: (tensor) predictions, sized [N,D].
          y: (tensor) targets, sized [N,].

        Return:
          (tensor) focal loss.
        '''
        gamma = 2.0
        r = torch.arange(x.size(0))
        ce = F.log_softmax(x, dim=1)[r, y]
        pt = torch.exp(ce)
        weights = (1-pt).pow(gamma)
        loss = -(weights * ce)

        return reduce(loss, reduction)

    def _sigmoid_focal_loss(self, pred, target, reduction='none'):
        alpha = 0.25
        gamma = 2.0
        pred_sigmoid = pred.sigmoid()
        target = torch.eye(self.num_classes, device=pred.device, dtype=pred.dtype)[target]

        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) *
                        (1 - target)) * pt.pow(gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight
        loss = loss.sum(dim=-1)
        loss = reduce(loss, reduction)
        return loss

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [N, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [N, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [N, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [N, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(cls_preds, cls_targets).
        '''
        mask_ign = cls_targets < 0
        cls_targets[mask_ign] = 0

        pos = cls_targets > 0  # [N,#anchors]
        batch_size = pos.size(0)
        num_pos = pos.sum().item()

        #===============================================================
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        #===============================================================
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask], reduction='sum')
        #===============================================================
        # cls_loss = CrossEntropyLoss(cls_preds, cls_targets)
        #===============================================================
        if self.mode != 'focal':
            cls_loss = F.cross_entropy(cls_preds.view(-1, self.num_classes), \
                                       cls_targets.view(-1), reduction='none')  # [N*#anchors,]
            cls_loss = cls_loss.view(batch_size, -1)
            cls_loss[mask_ign] = 0  # set ignored loss to 0
            if self.mode == 'hardneg':
                neg = self._hard_negative_mining(cls_loss, pos)  # [N,#anchors]
                cls_loss = cls_loss[pos|neg].sum()
            else:
                cls_loss = cls_loss.sum()
            cls_loss /= num_pos
        else:
            cls_loss = self.focal_loss(cls_preds.view(-1, self.num_classes), cls_targets.view(-1))
            cls_loss = cls_loss.view(batch_size, -1)
            cls_loss[mask_ign] = 0
            cls_loss = cls_loss.sum()
            cls_loss /= num_pos

        #print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.item()/num_pos, cls_loss.item()/num_pos), end=' | ')
        loc_loss /= num_pos
        return loc_loss, cls_loss