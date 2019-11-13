"""
Tests anchors are working correctly.
"""
from __future__ import print_function
from core.anchors import Anchors
from core.utils import box, opts
from datasets.moving_box_detection import SquaresVideos
import torch
import pytest


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
        self.box_generator = SquaresVideos(self.batchsize, self.time, self.height, self.width, max_objects=3, max_classes=3, render=False)
        self.box_coder = Anchors(allow_low_quality_matches=allow_low_quality_matches, bg_iou_threshold=bg_iou_threshold, fg_iou_threshold=fg_iou_threshold)
        self.fmaps = []
        for i in range(len(self.box_coder.pyramid_levels)):
            self.fmaps += [torch.zeros((self.batchsize, 1, self.height >> (3 + i), self.width >> (3 + i)))]

    def abs_diff(self, x, y):
        diff = (x - y).abs()
        return diff, diff.max().item()

    def cat_diff(self, x, y):
        u1 = torch.unique(x, return_counts=True)[1]
        u2 = torch.unique(y, return_counts=True)[1]
        return u1-u2

    def encode_sequential(self, targets, anchors, anchors_xyxy, fg_iou_threshold, bg_iou_threshold, allow_low_quality_matches):
        gt_padded = box.pack_boxes_list_of_list(targets)
        all_loc, all_cls = [], []
        for t in range(len(gt_padded)):
            gt_boxes, gt_labels = gt_padded[t, :, :4], gt_padded[t, :, -1]
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
        gt_padded = box.pack_boxes_list_of_list(targets)
        gt_boxes = gt_padded[..., :4]
        batch_iou = box.batch_box_iou(anchors_xyxy, gt_boxes.clone())
        for t in range(len(gt_padded)):
            iou_t = box.box_iou(anchors_xyxy, gt_boxes[t])
            max_abs_diff = self.abs_diff(batch_iou[t], iou_t)[1]
            assert max_abs_diff == 0

    def pytestcase_batched_encode_no_low_quality_with_dummies_encoded(self):
        self.init(3, 7, allow_low_quality_matches=False)
        targets = [self.box_generator[i][1] for i in range(self.batchsize)]
        targets = [[targets[i][t] for i in range(self.batchsize)] for t in range(self.time)]
        anchors, anchors_xyxy = self.box_coder(self.fmaps)
        # With dummies encoded
        loc_targets, cls_targets = self.box_coder.encode(self.fmaps, targets)

        loc_targets2, cls_targets2 = self.encode_sequential(targets, anchors, anchors_xyxy,
                                                            self.box_coder.fg_iou_threshold,
                                                            self.box_coder.bg_iou_threshold,
                                                            self.box_coder.allow_low_quality_matches)

        loc_diff, max_loc_diff = self.abs_diff(loc_targets, loc_targets2)
        cls_diff, max_cls_diff = self.abs_diff(cls_targets, cls_targets2)

        cat_diff = self.cat_diff(cls_targets, cls_targets2)

        assert max_loc_diff == 0
        assert max_cls_diff == 0
        assert cat_diff.abs().max() == 0

    def pytestcase_batched_encode_allow_quality_with_dummies_encoded(self):
        self.init(3, 7, allow_low_quality_matches=True)
        targets = [self.box_generator[i][1] for i in range(self.batchsize)]
        targets = [[targets[i][t] for i in range(self.batchsize)] for t in range(self.time)]
        anchors, anchors_xyxy = self.box_coder(self.fmaps)
        # With dummies encoded
        loc_targets, cls_targets = self.box_coder.encode(self.fmaps, targets)

        loc_targets2, cls_targets2 = self.encode_sequential(targets, anchors, anchors_xyxy,
                                                            self.box_coder.fg_iou_threshold,
                                                            self.box_coder.bg_iou_threshold,
                                                            self.box_coder.allow_low_quality_matches)

        loc_diff, max_loc_diff = self.abs_diff(loc_targets, loc_targets2)
        cls_diff, max_cls_diff = self.abs_diff(cls_targets, cls_targets2)

        cat_diff = self.cat_diff(cls_targets, cls_targets2)

        assert max_loc_diff == 0
        assert max_cls_diff == 0
        assert cat_diff.abs().max() == 0


    # TODO: enable this when debugging, you can see what step is failing
    # def pytestcase_batched_encode_step_by_step(self):
    #     # This checks every step of encoding
    #     #TODO: add the allow_low_quality_match step
    #     self.init(3, 7, allow_low_quality_matches=True)
    #     anchors, anchors_xyxy = self.box_coder(self.fmaps)
    #     targets = [self.box_generator[i][1] for i in range(self.batchsize)]
    #     targets = [[targets[i][t] for i in range(self.batchsize)] for t in range(self.time)]
    #     gt_padded = box.pack_boxes_list_of_list(targets)
    #     gt_boxes = gt_padded[..., :4]
    #     gt_labels = gt_padded[..., 4]
    #     batch_iou = box.batch_box_iou(anchors_xyxy, gt_boxes.clone())
    #     batch_best_target_per_prior, batch_best_target_per_prior_index = batch_iou.max(-1)  # [N, A]
    #
    #
    #     _, batch_best_prior_per_target_index = batch_iou.max(-2)  # [N, M]
    #     indices = []
    #     if self.box_coder.allow_low_quality_matches:
    #         max_size = batch_iou.shape[-1]
    #         for target_index in range(max_size):
    #             index = batch_best_prior_per_target_index[..., target_index:target_index + 1]
    #             indices.append(index)
    #             batch_best_target_per_prior_index.scatter_(-1, index, target_index)
    #             batch_best_target_per_prior.scatter_(-1, index, 2.0)
    #
    #     batch_mask_bg = (batch_best_target_per_prior < self.bg_iou_threshold)
    #     batch_mask_ign = (batch_best_target_per_prior > self.bg_iou_threshold) * (batch_best_target_per_prior < self.fg_iou_threshold)
    #
    #     batch_labels = torch.gather(gt_labels, 1, batch_best_target_per_prior_index)
    #     batch_labels[batch_mask_ign] = -1
    #     batch_labels[batch_mask_bg] = 0
    #
    #     index = batch_best_target_per_prior_index[..., None].expand(len(gt_padded), len(anchors), 4)
    #     batch_boxes = torch.gather(gt_boxes, 1, index)
    #     batch_loc_targets = box.bbox_to_deltas(batch_boxes, anchors[None])
    #     batch_cls_targets = batch_labels.view(-1, len(anchors))
    #
    #     # schizo-sanity
    #     # batch_loc_targets2, batch_cls_targets2 = self.box_coder.encode(self.fmaps, targets)
    #     # assert self.abs_diff(batch_loc_targets, batch_loc_targets2)[1] == 0
    #     # assert self.abs_diff(batch_cls_targets, batch_cls_targets2)[1] == 0
    #     #
    #     # seq_loc_targets, seq_cls_targets = self.encode_sequential(targets, anchors, anchors_xyxy, self.fg_iou_threshold,
    #     #                                                   self.bg_iou_threshold, self.allow_low_quality_matches)
    #     #
    #     # assert self.abs_diff(batch_loc_targets, seq_loc_targets)[1] == 0
    #     # assert self.abs_diff(batch_cls_targets, seq_cls_targets)[1] == 0
    #
    #     for t in range(len(gt_padded)):
    #         # same best_target_per_prior
    #         best_target_per_prior, best_target_per_prior_index = batch_iou[t].max(-1)
    #
    #         if self.box_coder.allow_low_quality_matches:
    #             _, best_prior_per_target_index = batch_iou[t].max(0)
    #
    #             assert self.abs_diff(batch_best_prior_per_target_index[t], best_prior_per_target_index)[1] == 0
    #
    #             for target_index, prior_index in enumerate(best_prior_per_target_index):
    #                 best_target_per_prior_index[prior_index] = target_index
    #             # 2.0 is used to make sure every target has a prior assigned
    #             best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    #
    #
    #         val_abs_diff = self.abs_diff(batch_best_target_per_prior[t], best_target_per_prior)[1]
    #         index_abs_diff = self.abs_diff(batch_best_target_per_prior_index[t], best_target_per_prior_index)[1]
    #
    #         assert val_abs_diff == 0
    #         assert index_abs_diff == 0


            # # same mask
            # mask_bg = (best_target_per_prior < self.bg_iou_threshold)
            # mask_ign = (best_target_per_prior > self.bg_iou_threshold) * (best_target_per_prior < self.fg_iou_threshold)
            # bg_diff = self.abs_diff(batch_mask_bg[t].float(), mask_bg.float())[1]
            # ign_diff = self.abs_diff(batch_mask_ign[t].float(), mask_ign.float())[1]
            # assert bg_diff == 0
            # assert ign_diff == 0
            #
            # # same label
            # labels = gt_labels[t][best_target_per_prior_index]
            # labels[mask_ign] = -1
            # labels[mask_bg] = 0
            # label_diff = self.abs_diff(batch_labels[t], labels)[1]
            # assert label_diff == 0
            #
            # #same encoded box
            # boxes = gt_boxes[t][best_target_per_prior_index]
            # boxes_diff = self.abs_diff(batch_boxes[t], boxes)[1]
            # assert boxes_diff == 0


            # loc_targets = box.bbox_to_deltas(boxes, anchors)
            # cls_targets = labels
            #
            # assert self.abs_diff(seq_loc_targets[t], loc_targets)[1] == 0
            # assert self.abs_diff(seq_cls_targets[t], cls_targets)[1] == 0