from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.models as models


class BackBone(nn.Module):
    def __init__(self, module, frozen_stages=-1, norm_eval=False):
        super(BackBone, self).__init__()
        self.features = nn.Sequential()
        self.frozen_stages = frozen_stages
        self.copy_features(module)
        self.norm_eval = norm_eval
        self.outputs = []
        self.out_channel_list = []
        self.add_collect_hooks()

    def forward(self, x):
        self.outputs = []
        self.features(x)
        return self.outputs

    def copy_features(self, module):
        raise NotImplementedError()

    def add_collect_hooks(self):
        raise NotImplementedError()

    def collect_hook(self, m, x, y):
        self.outputs.append(y)

    def train(self, mode=True):
        super(BackBone, self).train(mode)
        self.freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def freeze_stages(self):
        for i in range(self.frozen_stages):
            m = self.features[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False


class ResNet(BackBone):
    def __init__(self, in_channels, module, frozen_stages=-1, norm_eval=False):
        super(ResNet, self).__init__(module, frozen_stages, norm_eval)

        if in_channels != 3:
            self.features[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.frozen_stages = max(self.frozen_stages, 4)
            self.norm_eval = False

    def copy_features(self, module):
        self.features = nn.Sequential(module.conv1,
                                      module.bn1,
                                      module.relu,
                                      module.maxpool,
                                      module.layer1,
                                      module.layer2,
                                      module.layer3,
                                      module.layer4
                                      )

    def add_collect_hooks(self):
        self.features[5].register_forward_hook(self.collect_hook)
        self.features[6].register_forward_hook(self.collect_hook)
        self.features[7].register_forward_hook(self.collect_hook)

        self.out_channel_list = [self.features[5][-1].conv3.out_channels,
                                 self.features[6][-1].conv3.out_channels,
                                 self.features[7][-1].conv3.out_channels
                                 ]


class MobileNet(BackBone):
    def __init__(self, in_channels, pretrained=True, frozen_stages=-1, norm_eval=False):
        super(MobileNet, self).__init__(models.mobilenet_v2(pretrained=pretrained), frozen_stages, norm_eval)

        if in_channels != 3:
            self.features[0] = models.mobilenet.ConvBNReLU(in_channels, 32, stride=2)
            self.frozen_stages = max(self.frozen_stages, 1)
            self.norm_eval = False

    def get_stride(self, layer):
        if isinstance(layer, models.mobilenet.ConvBNReLU):
            s = layer[0].stride
        elif isinstance(layer, models.mobilenet.InvertedResidual):
            s = layer.stride
        return s[0] if isinstance(s, tuple) else s

    def add_collect_hooks(self):
        stride = 1
        for i, feature in enumerate(self.features):
            s = self.get_stride(feature)
            if s == 2:
                if stride >= 8:
                    self.features[i - 1].register_forward_hook(self.collect_hook)
                    self.out_channel_list.append(self.features[i - 1].conv[-2].out_channels)
                stride *= 2
        self.features[-1].register_forward_hook(self.collect_hook)
        self.out_channel_list.append(self.features[-1].conv[-2].out_channels)

    def copy_features(self, module):
        self.features = module.features[:-2]


def resnet18(in_channels, pretrained, frozen_stages=-1):
    return ResNet(in_channels, models.resnet18(pretrained=pretrained), frozen_stages)


def resnet34(in_channels, pretrained, frozen_stages=-1):
    return ResNet(in_channels, models.resnet34(pretrained=pretrained), frozen_stages)


def resnet50(in_channels, pretrained, frozen_stages=-1):
    return ResNet(in_channels, models.resnet50(pretrained=pretrained), frozen_stages)



if __name__ == '__main__':
    x = torch.rand(5, 2, 128, 128)
    net = ResNet(x.size(1), models.resnet18(pretrained=True))
    y = net(x)
    print('resnet: ', net.out_channel_list)
    print(len(y), [item.shape for item in y])

    net = MobileNet(x.size(1), models.mobilenet_v2(pretrained=True))
    print('mobilenet: ', net.out_channel_list)
    y = net(x)
    print(len(y), [item.shape for item in y])