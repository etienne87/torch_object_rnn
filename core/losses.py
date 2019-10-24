from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


def reduce(loss, mode='none'):
    ''' reduce mode

    :param loss:
    :param mode:
    :return:
    '''
    if mode == 'mean':
        loss = loss.mean()
    elif mode == 'sum':
        loss = loss.sum()
    return loss


def softmax_focal_loss(x, y, reduction='none'):
    ''' softmax focal loss

    :param x: [N, A, C+1]
    :param y: [N, A]
    :param reduction:
    :return:
    '''
    gamma = 2.0
    r = torch.arange(x.size(0))
    ce = F.log_softmax(x, dim=1)[r, y]
    pt = torch.exp(ce)
    weights = (1-pt).pow(gamma)
    loss = -(weights * ce)
    loss[y < 0] = 0
    return reduce(loss, reduction)


def sigmoid_focal_loss(x, y, reduction='none'):
    ''' sigmoid focal loss

    :param x: [N, A, C]
    :param y: [N, A] (-1: ignore, 0: background, [1,C+1]: classes)
    :param reduction:
    :return:
    '''
    alpha = 0.25
    gamma = 2.0
    s = x.sigmoid()
    batchsize, num_anchors, num_classes = x.shape

    y2 = y.unsqueeze(2)
    fg = (y2>0).float()
    y_index = (y2 - 1).clamp_(0)
    t = torch.zeros((len(x), num_anchors, num_classes), dtype=x.dtype, device=x.device)
    t.scatter_(2, y_index, fg)

    pt = (1 - s) * t + s * (1 - t)
    # focal_weight = (alpha * t + (1 - alpha) *
    #                 (1 - t)) * pt.pow(gamma)

    focal_weight = pt.pow(gamma)

    loss = F.binary_cross_entropy_with_logits(
        x, t, reduction='none') * focal_weight
    loss = loss.sum(dim=-1)
    loss[y < 0] = 0

    loss = reduce(loss, reduction)
    return loss


def ohem_loss(cls_preds, cls_targets, pos, batchsize):
    ''' hard-negative mining

    :param cls_preds:
    :param pos:
    :param neg:
    :return:
    '''
    cls_loss = F.cross_entropy(cls_preds.view(-1, cls_preds.size(-1)),
                               cls_targets.view(-1), ignore_index=-1, reduction='none')
    cls_loss = cls_loss.view(batchsize, -1)

    cls_loss2 = cls_loss * (pos.float() - 1)
    _, idx = cls_loss2.sort(1)  # sort by negative losses
    _, rank = idx.sort(1)  # [N,#anchors]
    num_neg = 3 * pos.sum(1)  # [N,]
    neg = rank < num_neg[:, None]
    cls_loss = cls_loss[pos | neg].sum()
    return cls_loss



class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        pos = cls_targets > 0
        num_pos = pos.sum().item()
        cls_loss = sigmoid_focal_loss(cls_preds, cls_targets, 'sum') / num_pos

        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,4]
        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask], reduction='sum') / num_pos

        return loc_loss, cls_loss
