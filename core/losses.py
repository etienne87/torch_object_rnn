from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
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

def cast(x, dtype):
    ''' cast in float or float16

    :param x: tensor to cast
    :param dtype: dst dtype
    '''
    if x.dtype == torch.float:
        return x.float()
    elif x.dtype == torch.float16:
        return x.half()


def softmax_focal_loss(x, y, reduction='none'):
    ''' softmax focal loss

    :param x: [N, A, C+1]
    :param y: [N, A]  (-1: ignore, 0: background, [1,C]: classes)
    :param reduction:
    :return:
    '''
    alpha = 0.25
    gamma = 2.0
    num_classes = x.size(-1)
    x = x.view(-1, num_classes)
    y = y.view(-1)
    r = torch.arange(x.size(0))
    ce = F.log_softmax(x, dim=-1)[r, y.clamp_(0)]
    pt = torch.exp(ce)
    weights = (1-pt).pow(gamma)

    # alpha version
    # p = y > 0
    # weights = (alpha * p + (1 - alpha) * (1 - p)) * weights.pow(gamma)

    loss = -(weights * ce)
    loss[y < 0] = 0
    return reduce(loss, reduction)


def sigmoid_focal_loss(x, y, reduction='none'):
    ''' sigmoid focal loss

    :param x: [N, A, C]
    :param y: [N, A] (-1: ignore, 0: background, [1,C]: classes)
    :param reduction:
    :return:
    '''
    alpha = 0.25
    gamma = 2.0
    s = x.sigmoid()
    batchsize, num_anchors, num_classes = x.shape
    

    y2 = y.unsqueeze(2)
    fg = (y2>0).to(x)
    y_index = (y2 - 1).clamp_(0)
    t = torch.zeros((len(x), num_anchors, num_classes), dtype=x.dtype, device=x.device)
    t.scatter_(2, y_index, fg)

    pt = (1 - s) * t + s * (1 - t)
    focal_weight = (alpha * t + (1 - alpha) *
                    (1 - t)) * pt.pow(gamma)

    # focal_weight = pt.pow(gamma)

    loss = F.binary_cross_entropy_with_logits(
        x, t, reduction='none') * focal_weight
    loss = loss.sum(dim=-1)
    loss[y < 0] = 0

    loss = reduce(loss, reduction)
    return loss


def softmax_ohem_loss(cls_preds, cls_targets, reduction='none'):
    ''' hard-negative mining

    :param cls_preds:
    :param pos:
    :param neg:
    :return:
    '''
    pos = cls_targets > 0
    batchsize = cls_preds.shape[0]
    cls_loss = F.cross_entropy(cls_preds.view(-1, cls_preds.size(-1)),
                               cls_targets.view(-1), ignore_index=-1, reduction='none')
    cls_loss = cls_loss.view(batchsize, -1)

    cls_loss2 = cls_loss * (pos.to(cls_preds) - 1)
    _, idx = cls_loss2.sort(1)  # sort by negative losses
    _, rank = idx.sort(1)  # [N,#anchors]
    num_neg = 3 * pos.sum(1)  # [N,]
    neg = rank < num_neg[:, None]
    cls_loss = reduce(cls_loss[pos | neg], reduction)
    return cls_loss


def smooth_l1_loss(pred, target, beta=0.11, reduction='sum'):
    """ smooth l1 loss
def cast(x, dtype):
    if x.dtype() == torch.float:
        return x.float()
    elif x.dtype() == torch.float16:
        return x.half()
    """
    x = (pred - target).abs()
    l1 = x - 0.5 * beta
    l2 = 0.5 * x ** 2 / beta
    reg_loss = torch.where(x >= beta, l1, l2)
    return reduce(reg_loss, reduction)



class DetectionLoss(nn.Module):
    def __init__(self, cls_loss_func='softmax_focal_loss'):
        super(DetectionLoss, self).__init__()
        self.cls_loss_func = getattr(sys.modules[__name__], cls_loss_func)
        self.reg_loss_func = F.smooth_l1_loss 
        print('cls function: ', self.cls_loss_func)
        print('reg function: ', self.reg_loss_func)

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        pos = cls_targets > 0
        num_pos = pos.sum().item()
        cls_loss = self.cls_loss_func(cls_preds, cls_targets, 'sum') / num_pos

        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,4]
        # loc_targets = cast(loc_targets[mask], loc_preds.dtype)
        loc_loss = self.reg_loss_func(loc_preds[mask], loc_targets[mask].to(loc_preds), reduction='sum') / num_pos
        return loc_loss, cls_loss
