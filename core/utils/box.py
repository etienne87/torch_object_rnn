import torch
import numpy as np
from numba import jit


def change_box_order(boxes, order):
    '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).

    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

    Returns:
      (tensor) converted bounding boxes, sized [N,4].
    '''
    assert order in ['xyxy2xywh','xywh2xyxy']
    a = boxes[...,:2]
    b = boxes[...,2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a+b)/2,b-a], -1)
    return torch.cat([a-b/2,a+b/2], -1)


def box_clamp(boxes, xmin, ymin, xmax, ymax):
    '''Clamp boxes.

    Args:
      boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [N,4].
      xmin: (number) min value of x.
      ymin: (number) min value of y.
      xmax: (number) max value of x.
      ymax: (number) max value of y.

    Returns:
      (tensor) clamped boxes.
    '''
    boxes[:,0].clamp_(min=xmin, max=xmax)
    boxes[:,1].clamp_(min=ymin, max=ymax)
    boxes[:,2].clamp_(min=xmin, max=xmax)
    boxes[:,3].clamp_(min=ymin, max=ymax)
    return boxes

def box_select(boxes, xmin, ymin, xmax, ymax):
    '''Select boxes in range (xmin,ymin,xmax,ymax).

    Args:
      boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [N,4].
      xmin: (number) min value of x.
      ymin: (number) min value of y.
      xmax: (number) max value of x.
      ymax: (number) max value of y.

    Returns:
      (tensor) selected boxes, sized [M,4].
      (tensor) selected mask, sized [N,].
    '''
    mask = (boxes[:,0]>=xmin) & (boxes[:,1]>=ymin) \
         & (boxes[:,2]<=xmax) & (boxes[:,3]<=ymax)
    boxes = boxes[mask,:]
    return boxes, mask

def box_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes.

    The box order must be (xmin, ymin, xmax, ymax).

    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou


def batch_box_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes.

    The box order must be (xmin, ymin, xmax, ymax).

    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [B,M,4].

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    lt = torch.max(box1[None,:,None,:2], box2[:,None,:,:2])  # [B,N,M,2] broadcast_max( (_,N,_,2), (B,_,M,2) )
    rb = torch.min(box1[None,:,None,2:], box2[:,None,:,2:])  # [B,N,M,2]

    wh = (rb-lt).clamp(min=0)      # [B,N,M,2]
    inter = wh[...,0] * wh[...,1]  # [B,N,M]

    area1 = (box1[...,2]-box1[...,0]) * (box1[...,3]-box1[...,1])  # [N,]
    area2 = (box2[...,2]-box2[...,0]) * (box2[...,3]-box2[...,1])  # [B,M,]
    iou = inter / (area1[None,:,None] + area2[:,None,:] - inter) # [B,N,M]
    return iou



def box_nms(bboxes, scores, threshold=0.5):
    '''Non maximum suppression.

    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) confidence scores, sized [N,].
      threshold: (float) overlap threshold.

    Returns:
      keep: (tensor) selected indices.

    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1) * (y2-y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        if order.dim() == 0:
            i = order.item()
        else:
            i = order[0]
    
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w * h

        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (overlap<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.tensor(keep, dtype=torch.long)

def box_soft_nms(boxes, scores, nms_threshold=0.3,
                                 soft_threshold=0.3,
                                 sigma=0.5,
                                 mode='union'):
    """
    soft-nms implentation according the soft-nms paper
    :param bboxes:
    :param scores:
    :param labels:
    :param nms_threshold:
    :param soft_threshold:
    :return:
    """

    keep = []
    c_boxes = boxes
    weights = scores.clone()
    x1 = c_boxes[:, 0]
    y1 = c_boxes[:, 1]
    x2 = c_boxes[:, 2]
    y2 = c_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = weights.sort(0, descending=True)
    while order.numel() > 0:
        if order.dim() == 0:
            i = order.item()
        else:
            i = order[0]

        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids_t= (ovr>=nms_threshold).nonzero().squeeze()

        weights[[order[ids_t+1]]] *= torch.exp(-(ovr[ids_t] * ovr[ids_t]) / sigma)

        ids = (weights[order[1:]] >= soft_threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        c_boxes = c_boxes[order[1:]][ids]
        _, order = weights[order[1:]][ids].sort(0, descending=True)
        if c_boxes.dim()==1:
            c_boxes=c_boxes.unsqueeze(0)
        x1 = c_boxes[:, 0]
        y1 = c_boxes[:, 1]
        x2 = c_boxes[:, 2]
        y2 = c_boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    return torch.tensor(keep, dtype=torch.long)

@jit(nopython=True)
def np_box_nms(bboxes, scores, threshold=0.5):
    '''Non maximum suppression.

    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) confidence scores, sized [N,].
      threshold: (float) overlap threshold.

    Returns:
      keep: (tensor) selected indices.

    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return torch.tensor(keep, dtype=torch.long)


def bbox_to_deltas(boxes, default_boxes, variances=[0.1, 0.2], max_width=10000):
    boxes = change_box_order(boxes, "xyxy2xywh")
    boxes[..., 2:] = boxes[..., 2:].clamp_(2, max_width)
    loc_xy = (boxes[..., :2] - default_boxes[..., :2]) / default_boxes[..., 2:] / variances[0]
    loc_wh = torch.log(boxes[..., 2:] / default_boxes[..., 2:]) /variances[1]
    deltas = torch.cat([loc_xy, loc_wh], -1)
    return deltas


def deltas_to_bbox(loc_preds, default_boxes, variances=[0.1, 0.2]):
    xy = loc_preds[..., :2] * variances[0] * default_boxes[..., 2:] + default_boxes[..., :2]
    wh = torch.exp(loc_preds[..., 2:] * variances[1]) * default_boxes[..., 2:]
    box_preds = torch.cat([xy - wh / 2, xy + wh / 2], -1)
    return box_preds


def pack_boxes_list_of_list(targets, label_offset=1):
    tbins, batchsize = len(targets), len(targets[0])
    max_size = max([max([len(frame) for frame in time]) for time in targets])
    max_size = max(2, max_size)
    gt_padded = torch.ones((tbins, batchsize, max_size, 5), dtype=torch.float32) * -1
    for t in range(len(targets)):
        for i in range(len(targets[t])):
            frame = targets[t][i]
            gt_padded[t, i, :len(frame)] = frame
            gt_padded[t, i, :len(frame), 4] += label_offset
    return gt_padded.view(-1, max_size, 5)


def pack_boxes_list(targets, label_offset=1):
    batchsize = len(targets)
    max_size = max([len(frame) for frame in targets])
    max_size = max(2, max_size)
    gt_padded = torch.ones((batchsize, max_size, 5), dtype=torch.float32) * -1
    for t in range(len(targets)):
        frame = targets[t][i]
        gt_padded[t, :len(frame)] = frame
        gt_padded[t, :len(frame), 4] += label_offset
    return gt_padded.view(-1, max_size, 5)


def assign_priors(gt_boxes, gt_labels, corner_form_priors,
                  fg_iou_threshold, bg_iou_threshold, allow_low_quality_matches=True):
    """Assign ground truth boxes and targets to priors.
    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priors): labels for priors.
    """
    # size: num_priors x num_targets
    ious = box_iou(corner_form_priors, gt_boxes)
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    if allow_low_quality_matches:
        for target_index, prior_index in enumerate(best_prior_per_target_index):
            best_target_per_prior_index[prior_index] = target_index
        # 2.0 is used to make sure every target has a prior assigned
        best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)

    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]

    mask = (best_target_per_prior > bg_iou_threshold) * (best_target_per_prior < fg_iou_threshold)

    labels[mask] = -1
    labels[best_target_per_prior < bg_iou_threshold] = 0  # the background id
    boxes = gt_boxes[best_target_per_prior_index]


    return boxes, labels

def assign_priors_v2(gt_boxes, gt_labels, corner_form_priors,
                  fg_iou_threshold, bg_iou_threshold, min_pos_threshold=0.2):
    ious = box_iou(corner_form_priors, gt_boxes)
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)

    for target_index in range(len(gt_boxes)):
        best_prior_per_target_index = torch.argmax(ious[:, target_index])
        if ious[best_prior_per_target_index, target_index] >= min_pos_threshold:
            #lock the anchor for this gt
            best_target_per_prior_index[best_prior_per_target_index] = target_index
            ious[best_prior_per_target_index, :] = 0

    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]

    mask = (best_target_per_prior > bg_iou_threshold) * (best_target_per_prior < fg_iou_threshold)

    labels[mask] = -1
    labels[best_target_per_prior < bg_iou_threshold] = 0  # the background id
    boxes = gt_boxes[best_target_per_prior_index]

    return boxes, labels
