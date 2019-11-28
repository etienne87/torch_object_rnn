"""
Tests anchors are working correctly.
"""
from __future__ import print_function
from core.anchors import Anchors
from core.utils import box, opts
from datasets.moving_box_detection import SquaresVideos
import torch


class TestAnchors(object):
    """
    test of box coder class.
    """
    def init(self, time=5, batchsize=3, height=256, width=256, allow_low_quality_matches=False, bg_iou_threshold=0.4, fg_iou_threshold=0.5):
        self.batchsize = batchsize
        self.time = time
        self.height, self.width = height, width
        self.bg_iou_threshold = bg_iou_threshold
        self.fg_iou_threshold = fg_iou_threshold
        self.allow_low_quality_matches = allow_low_quality_matches
        self.num_classes = 3
        self.box_generator = SquaresVideos(self.batchsize, self.time, self.height, self.width, max_objects=3, max_classes=3, render=False)
        self.box_coder = Anchors(allow_low_quality_matches=allow_low_quality_matches, bg_iou_threshold=bg_iou_threshold, fg_iou_threshold=fg_iou_threshold)
        self.fmaps = []
        for i in range(len(self.box_coder.pyramid_levels)):
            self.fmaps += [torch.zeros((self.batchsize, 1, self.height >> (3 + i), self.width >> (3 + i)))]

    def assert_equal(self, x, y):
        assert (x-y).abs().max().item() == 0

    def abs_diff(self, x, y):
        diff = (x - y).abs()
        return diff, diff.max().item()

    def cat_diff(self, x, y):
        u1 = torch.unique(x, return_counts=True)[1]
        u2 = torch.unique(y, return_counts=True)[1]
        return u1-u2

    def encode_sequential(self, targets, anchors, anchors_xyxy, fg_iou_threshold, bg_iou_threshold, allow_low_quality_matches, remove_dummies=False):
        gt_padded, sizes = box.pack_boxes_list_of_list(targets)
        all_loc, all_cls = [], []
        for t in range(len(gt_padded)):
            gt_boxes, gt_labels = gt_padded[t, :, :4], gt_padded[t, :, -1]
            if remove_dummies:
                max_size = sizes[t]
                gt_boxes, gt_labels = gt_boxes[:max_size], gt_labels[:max_size]
            boxes, cls_t = box.assign_priors(gt_boxes, gt_labels, anchors_xyxy, fg_iou_threshold, bg_iou_threshold, allow_low_quality_matches)
            loc_t = box.bbox_to_deltas(boxes, anchors)
            all_loc.append(loc_t.unsqueeze(0))
            all_cls.append(cls_t.unsqueeze(0).long())
        all_loc = torch.cat(all_loc, dim=0)  # (N,#anchors,4)
        all_cls = torch.cat(all_cls, dim=0)  # (N,#anchors,C)
        return all_loc, all_cls

    def pytestcase_batch_box_iou(self):
        self.init(7, 3)
        anchors, anchors_xyxy = self.box_coder(self.fmaps)
        targets = [self.box_generator[i][1] for i in range(self.batchsize)]
        targets = [[targets[i][t] for i in range(self.batchsize)] for t in range(self.time)]
        gt_padded, _ = box.pack_boxes_list_of_list(targets)
        gt_boxes = gt_padded[..., :4]
        batch_iou = box.batch_box_iou(anchors_xyxy, gt_boxes.clone())
        for t in range(len(gt_padded)):
            iou_t = box.box_iou(anchors_xyxy, gt_boxes[t])
            max_abs_diff = self.abs_diff(batch_iou[t], iou_t)[1]
            assert max_abs_diff == 0

    def pytestcase_batched_encode_only_best_quality(self):
        self.init(3, 7, allow_low_quality_matches=False)
        targets = [self.box_generator[i][1] for i in range(self.batchsize)]
        targets = [[targets[i][t] for i in range(self.batchsize)] for t in range(self.time)]
        anchors, anchors_xyxy = self.box_coder(self.fmaps)
        loc_targets, cls_targets = self.box_coder.encode(anchors, anchors_xyxy, targets)
        loc_targets2, cls_targets2 = self.encode_sequential(targets, anchors, anchors_xyxy,
                                                            self.box_coder.fg_iou_threshold,
                                                            self.box_coder.bg_iou_threshold,
                                                            self.box_coder.allow_low_quality_matches,
                                                            remove_dummies=True)
        loc_diff, max_loc_diff = self.abs_diff(loc_targets, loc_targets2)
        cls_diff, max_cls_diff = self.abs_diff(cls_targets, cls_targets2)
        cat_diff = self.cat_diff(cls_targets, cls_targets2)

        assert max_loc_diff == 0
        assert max_cls_diff == 0
        assert cat_diff.abs().max() == 0

    def pytestcase_batched_encode_allow_low_quality(self):
        self.init(3, 7, allow_low_quality_matches=True)
        targets = [self.box_generator[i][1] for i in range(self.batchsize)]
        targets = [[targets[i][t] for i in range(self.batchsize)] for t in range(self.time)]
        anchors, anchors_xyxy = self.box_coder(self.fmaps)
        loc_targets, cls_targets = self.box_coder.encode(anchors, anchors_xyxy, targets)
        loc_targets2, cls_targets2 = self.encode_sequential(targets, anchors, anchors_xyxy,
                                                            self.box_coder.fg_iou_threshold,
                                                            self.box_coder.bg_iou_threshold,
                                                            self.box_coder.allow_low_quality_matches,
                                                            remove_dummies=True)
        loc_diff, max_loc_diff = self.abs_diff(loc_targets, loc_targets2)
        cls_diff, max_cls_diff = self.abs_diff(cls_targets, cls_targets2)
        cat_diff = self.cat_diff(cls_targets, cls_targets2)

        assert max_loc_diff == 0
        assert max_cls_diff == 0
        assert cat_diff.abs().max() == 0

    def one_hot(self, y, num_classes):
        y2 = y.unsqueeze(2)
        fg = (y2 > 0).float()
        y_index = (y2 - 1).clamp_(0)
        t = torch.zeros((y.shape[0], y.shape[1], num_classes), dtype=torch.float)
        t.scatter_(2, y_index, fg)
        return t

    def pytestcase_batched_decode_boxes(self):
        self.init(1, 1, allow_low_quality_matches=True, bg_iou_threshold=0.4, fg_iou_threshold=0.5)
        targets = [self.box_generator[i][1] for i in range(self.batchsize)]
        targets = [[targets[i][t] for i in range(self.batchsize)] for t in range(self.time)]
        anchors, anchors_xyxy = self.box_coder(self.fmaps)
        loc_targets, cls_targets = self.box_coder.encode(anchors, anchors_xyxy, targets)
        scores = self.one_hot(cls_targets, self.num_classes)

        self.box_coder.decode_func = self.box_coder.decode_per_image

        boxes1 = self.box_coder.decode(anchors, loc_targets.clone(), scores, self.batchsize, 0.99)

        self.box_coder.decode_func = self.box_coder.batched_decode

        boxes2 = self.box_coder.decode(anchors, loc_targets, scores, self.batchsize, 0.99)

        for t in range(self.time):
            for x, y in zip(boxes1[t], boxes2[t]):
                b,s,l = x
                b2,s2,l2 = y

                self.assert_equal(b, b2)
                self.assert_equal(s, s2)
                self.assert_equal(l, l2)

    def pytestcase_all_gt_should_be_matched_even_low_iou(self):
        """
        Boxes with small iou should be matched
        :return:
        """
        box_coder = Anchors(allow_low_quality_matches=True)
        anchors_xyxy = torch.tensor([[25, 25, 250, 250],
                                                         [20, 20, 50, 50],
                                                         [3, 3, 4, 4]], dtype=torch.float32)
        anchors = box.change_box_order(anchors_xyxy, 'xyxy2xywh')

        targets = torch.tensor([[120, 120, 250, 250, 1], [20, 20, 22, 22, 2]], dtype=torch.float32)
        targets = [[targets]]

        _, cls_targets = box_coder.encode(anchors, anchors_xyxy, targets)
        assert len(torch.unique(cls_targets)) == 3   # first box and +1 is for background class