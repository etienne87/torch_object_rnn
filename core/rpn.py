from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.modules import Bottleneck, ConvLayer
import math


class FCHead(nn.Module):
    def __init__(self, in_channels, num_classes, act='sigmoid', n_layers=0):
        super(FCHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.box_head = self._make_head(in_channels, 4, n_layers)
        self.cls_head = self._make_head(in_channels, self.num_classes, n_layers)

    def _make_head(self, in_channels, out_channels, n_layers): 
        layers = [nn.Linear(in_channels, 256), nn.ReLU()]
        for i in range(n_layers):
            layers += [nn.Linear(256, 256), nn.ReLU()]
        layers += [nn.Linear(256, out_channels)]   
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        loc_preds = self.box_head(x)
        cls_preds = self.cls_head(x)
        return loc_preds, cls_preds


class BoxHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, act='sigmoid', n_layers=0):
        super(BoxHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_anchors = num_anchors

        self.aspect_ratios = []
        self.act = act

        conv_func = lambda x,y: ConvLayer(x, y, norm='none', activation='ReLU')
        self.box_head = self._make_head(in_channels, self.num_anchors * 4, n_layers, conv_func)
        self.cls_head = self._make_head(in_channels, self.num_anchors * self.num_classes, n_layers, conv_func)


        def initialize_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
        self.cls_head.apply(initialize_layer)
        self.box_head.apply(initialize_layer)

        if self.act == 'softmax':
            self.softmax_init(self.cls_head[-1])
        else:
            self.sigmoid_init(self.cls_head[-1])

    def sigmoid_init(self, l):
        print('rpn1')
        px = 0.99
        bias_bg = math.log(px / (1 - px))
        torch.nn.init.normal_(l.weight, std=0.01)
        torch.nn.init.constant_(l.bias, 0)
        l.bias.data = l.bias.data.reshape(self.num_anchors, self.num_classes)
        l.bias.data[:, 0:] -= bias_bg
        l.bias.data = l.bias.data.reshape(-1)

    def softmax_init(self, l):
        px = 0.99
        K = self.num_classes - 1
        bias_bg = math.log(K * px / (1 - px))
        torch.nn.init.normal_(l.weight, std=0.01)
        torch.nn.init.constant_(l.bias, 0)
        l.bias.data = l.bias.data.reshape(self.num_anchors, self.num_classes)
        l.bias.data[:, 0] += bias_bg
        l.bias.data = l.bias.data.reshape(-1)

    def _make_head(self, in_planes, out_planes, n_layers, conv_func):
        layers = []
        layers.append(conv_func(in_planes, 256))
        for _ in range(n_layers):
            layers.append(conv_func(256, 256))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def reset(self):
        self.feature_extractor.reset()

    def _apply_head(self, layer, xs, ndims):
        out = []
        for x in xs:
            y = layer(x).permute(0, 2, 3, 1).contiguous()
            y = y.view(y.size(0), -1, ndims)
            out.append(y)
        out = torch.cat(out, 1)
        return out

    def forward(self, xs):
        loc_preds = self._apply_head(self.box_head, xs, 4)
        cls_preds = self._apply_head(self.cls_head, xs, self.num_classes)
        return loc_preds, cls_preds  

    def probas(self, cls_preds):
        if not self.training:
            if self.act == 'softmax':
                cls_preds = F.softmax(cls_preds, dim=2)
            else:
                cls_preds = torch.sigmoid(cls_preds)
        return cls_preds


class SSDHead(nn.Module):
    def __init__(self, in_channels_list, num_anchors, num_classes, act='sigmoid'):
        super(SSDHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.aspect_ratios = []
        self.act = act

        conv_func = lambda x,y: nn.Conv2d(x, y, kernel_size=3, stride=1, padding=1)
        self.box_heads = nn.ModuleList()
        self.cls_heads = nn.ModuleList()
        for in_channels in in_channels_list:
            loc_head = conv_func(in_channels, self.num_anchors * 4)
            cls_head = conv_func(in_channels, self.num_anchors * self.num_classes)
            torch.nn.init.normal_(loc_head.weight, std=0.01)
            torch.nn.init.constant_(loc_head.bias, 0)
            self.box_heads.append(loc_head)
            self.cls_heads.append(cls_head)
        
    def reset(self):
        self.feature_extractor.reset()

    def _apply_head(self, layers, xs, ndims):
        out = []
        for x, layer in zip(xs, layers):
            y = layer(x).permute(0, 2, 3, 1).contiguous()
            y = y.view(y.size(0), -1, ndims)
            out.append(y)
        out = torch.cat(out, 1)
        return out

    def forward(self, xs):
        loc_preds = self._apply_head(self.box_heads, xs, 4)
        cls_preds = self._apply_head(self.cls_heads, xs, self.num_classes)

        if not self.training:
            if self.act == 'softmax':
                cls_preds = F.softmax(cls_preds, dim=2)
            else:
                cls_preds = torch.sigmoid(cls_preds)

        return loc_preds, cls_preds

    def probas(self, cls_preds):
        if not self.training:
            if self.act == 'softmax':
                cls_preds = F.softmax(cls_preds, dim=2)
            else:
                cls_preds = torch.sigmoid(cls_preds)
        return cls_preds