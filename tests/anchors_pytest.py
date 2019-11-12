"""
Tests anchors are working correctly.
"""
from __future__ import print_function
from core.anchors import Anchors
from core.utils import box
from datasets.moving_box_detection import SquaresVideos
import torch
import pytest


class TestAnchors(object):
    """
    test of box coder class.
    """

    def pytestcase_assign_priors_equivalence(self):

        batchsize = 3
        time = 5
        height, width = 256, 256
        box_generator = SquaresVideos(batchsize, time, height, width, max_classes=3, render=False)

        targets = [box_generator[i] for i in range(3)]

        box_coder = Anchors()

        fmaps = []
        for i in range(box_coder.levels):
            fmaps += [torch.zeros((batchsize, 1, height>>3, width>>3))]

        loc_targets, cls_targets = box_coder.encode(fmaps, targets)



        print(loc_targets, cls_targets)


