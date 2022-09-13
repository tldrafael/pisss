"""Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`

Attributes:

    _out_channels (list of int): specify number of channels for each encoder feature tensor
    _depth (int): specify number of stages in decoder (in other words number of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer for encoder (usually 3)

Methods:

    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
"""
from copy import deepcopy

import torch
import torch.nn as nn

from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from pretrainedmodels.models.torchvision_models import pretrained_settings

from ._base import EncoderMixin


def conv7x7(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3 * dilation, groups=groups, bias=False, dilation=dilation)


def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2 * dilation, groups=groups, bias=False, dilation=dilation)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class StemBlock1(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes[0])
        self.bn1 = nn.BatchNorm2d(planes[0])

        self.conv2 = conv5x5(inplanes, planes[1])
        self.bn2 = nn.BatchNorm2d(planes[1])

        self.conv3 = conv7x7(inplanes, planes[2])
        self.bn3 = nn.BatchNorm2d(planes[2])

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_1 = self.conv1(x)
        out_1 = self.bn1(out_1)

        out_2 = self.conv2(x)
        out_2 = self.bn2(out_2)

        out_3 = self.conv3(x)
        out_3 = self.bn3(out_3)

        out = torch.cat([out_1, out_2, out_3], dim=1)
        return self.relu(out)


class StemBlock2(nn.Module):
    def __init__(self, inplanes, planes, stride=2, channel_reduction=16, dilation=1):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes[0], stride=stride)
        self.bn1 = nn.BatchNorm2d(planes[0])

        self.conv2_1 = conv1x1(inplanes, channel_reduction)
        self.conv2_2 = conv3x3(channel_reduction, planes[1], stride=stride)
        self.bn2 = nn.BatchNorm2d(planes[1])

        self.conv3_1 = conv1x1(inplanes, channel_reduction)
        self.conv3_2 = conv5x5(channel_reduction, planes[2], stride=stride)
        self.bn3 = nn.BatchNorm2d(planes[2])

        self.conv4_1 = conv1x1(inplanes, channel_reduction)
        self.conv4_2 = conv7x7(channel_reduction, planes[3], stride=stride)
        self.bn4 = nn.BatchNorm2d(planes[3])

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        self.conv5 = conv1x1(inplanes, planes[4])
        self.bn5 = nn.BatchNorm2d(planes[4])

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = nn.Sequential(
                            conv1x1(inplanes, sum(planes), stride=stride),
                            nn.BatchNorm2d(sum(planes))
                            )

        self.conv6 = conv1x1(sum(planes), 64, stride=1)
        self.bn6 = nn.BatchNorm2d(64)

    def forward(self, x):
        identity = x

        out_1 = self.conv1(x)
        out_1 = self.bn1(out_1)

        out_2 = self.conv2_1(x)
        out_2 = self.conv2_2(out_2)
        out_2 = self.bn2(out_2)

        out_3 = self.conv3_1(x)
        out_3 = self.conv3_2(out_3)
        out_3 = self.bn3(out_3)

        out_4 = self.conv4_1(x)
        out_4 = self.conv4_2(out_4)
        out_4 = self.bn4(out_4)

        out_5 = self.maxpool(x)
        out_5 = self.conv5(out_5)
        out_5 = self.bn5(out_5)

        out = torch.cat([out_1, out_2, out_3, out_4, out_5], dim=1)
        out += self.downsample(identity)
        out = self.relu(out)

        out = self.conv6(out)
        out = self.bn6(out)
        return self.relu(out)


class RichStem(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.block1 = StemBlock1(3, [24, 24, 24])
        self.block2 = StemBlock2(72, [32, 64, 64, 64, 32], channel_reduction=16)

    def forward(self, x):
        out = self.block1(x)
        return self.block2(out)


class ClassicStem(nn.Module):
    def __init__(self, inplanes=64, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return self.relu(x)


class ParallelStem(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.classic = ClassicStem()
        self.rich = RichStem()

    def forward(self, x):
        return self.classic(x) + self.rich(x)


class ResNetEncoder(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, fl_maxpool=True, fl_richstem=False,
                 fl_parallelstem=False, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        self.fl_maxpool = fl_maxpool

        assert not fl_richstem or not fl_parallelstem, \
               "Or set fl_richstem or fl_richstem_parallel, but not both"
        self.fl_richstem = fl_richstem
        self.fl_parallelstem = fl_parallelstem

        if fl_parallelstem:
            self.stem = ParallelStem()
        else:
            self.stem = RichStem() if fl_richstem else ClassicStem()

        del self.conv1
        del self.bn1
        del self.relu

        del self.fc
        del self.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            self.stem,
            nn.Sequential(self.maxpool, self.layer1) if self.fl_maxpool else self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)

        if not self.fl_richstem and not self.fl_parallelstem:
            prefix = 'stem.'
        elif self.fl_parallelstem:
            prefix = 'stem.classic.'
        else:
            prefix = ''

        update_lnames = ['conv1.weight', 'bn1.weight', 'bn1.bias',
                         'bn1.running_mean', 'bn1.running_var']
        for k in list(state_dict.keys()):
            if k in update_lnames:
                state_dict['{}{}'.format(prefix, k)] = state_dict.pop(k)

        super().load_state_dict(state_dict, strict=False, **kwargs)


new_settings = {
    "resnet18": {
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth",  # noqa
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth",  # noqa
    },
    "resnet50": {
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth",  # noqa
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth",  # noqa
    },
    "resnext50_32x4d": {
        "imagenet": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext50_32x4-ddb3e555.pth",  # noqa
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext50_32x4-72679e44.pth",  # noqa
    },
    "resnext101_32x4d": {
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x4-dc43570a.pth",  # noqa
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x4-3f87e46b.pth",  # noqa
    },
    "resnext101_32x8d": {
        "imagenet": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth",
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x8-2cfe2f8b.pth",  # noqa
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x8-b4712904.pth",  # noqa
    },
    "resnext101_32x16d": {
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth",
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x16-15fffa57.pth",  # noqa
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x16-f3559a9c.pth",  # noqa
    },
    "resnext101_32x32d": {
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth",
    },
    "resnext101_32x48d": {
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth",
    },
}

pretrained_settings = deepcopy(pretrained_settings)
for model_name, sources in new_settings.items():
    if model_name not in pretrained_settings:
        pretrained_settings[model_name] = {}

    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }


resnet_encoders = {
    "resnet18": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet18"],
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [2, 2, 2, 2],
        },
    },
    "resnet34": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet34"],
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet50": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet50"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet101": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet101"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
        },
    },
    "resnet152": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet152"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 8, 36, 3],
        },
    },
    "resnext50_32x4d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext50_32x4d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x4d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x4d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x8d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x8d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 8,
        },
    },
    "resnext101_32x16d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x16d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 16,
        },
    },
    "resnext101_32x32d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x32d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 32,
        },
    },
    "resnext101_32x48d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x48d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 48,
        },
    },
}
