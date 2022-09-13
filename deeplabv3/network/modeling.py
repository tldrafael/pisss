from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import resnet
from .backbone import mobilenetv2
from .backbone import hrnetv2

def _segm_hrnet(name, backbone_name, num_classes, pretrained_backbone, **kwargs):

    backbone = hrnetv2.__dict__[backbone_name](pretrained_backbone)
    # HRNetV2 config:
    # the final output channels is dependent on highest resolution channel config (c).
    # output of backbone will be the inplanes to assp:
    hrnet_channels = int(backbone_name.split('_')[-1])
    inplanes = sum([hrnet_channels * 2 ** i for i in range(4)])
    low_level_planes = 256 # all hrnet version channel output from bottleneck is the same
    aspp_dilate = [12, 24, 36] # If follow paper trend, can put [24, 48, 72].

    if name=='deeplabv3plus':
        return_layers = {'stage4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'stage4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers, hrnet_flag=True)
    model = DeepLabV3(backbone, classifier)
    return model

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone, **kwargs):

    fl_maxpool = kwargs.get('fl_maxpool')
    fl_stemstride = kwargs.get('fl_stemstride')
    output_stride_lowlevel = 4 // max(2 * ((not fl_maxpool) + (not fl_stemstride)), 1)
    assert output_stride >= output_stride_lowlevel, 'Final output stride should be at least equal to the OS_lowlevel'
    output_stride_diff = output_stride // output_stride_lowlevel
    assert output_stride % output_stride_lowlevel == 0, 'The ratio between output_stride low-level and high-level should be an int'

    if not fl_stemstride and not fl_maxpool and output_stride > 8:
        raise ValueError('When avoiding StemStride and MaxPooling, the maximum possible stride is 8')
    elif ((fl_stemstride and not fl_maxpool) or (not fl_stemstride and fl_maxpool)) and output_stride > 16:
        raise ValueError('When avoiding MaxPooling, the maximum possible stride is 16')

    if output_stride_diff == 1:
        replace_stride_with_dilation=[True, True, True]
        aspp_dilate = [3, 6, 9]
    elif output_stride_diff == 2:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [6, 12, 18]
    elif output_stride_diff == 4:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [12, 24, 36]
    elif output_stride_diff == 8:
        replace_stride_with_dilation=[False, False, False]
        aspp_dilate = [24, 48, 72]
    else:
        raise ValueError('output_stride_diff must be at most equal 8')

    kwargs['output_stride_lowlevel'] = output_stride_lowlevel
    kwargs['output_stride_diff'] = output_stride_diff

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation, **kwargs)

    if backbone_name == 'resnet34':
        inplanes = 512
        low_level_planes = 64
    else:
        inplanes = 2048
        low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate, **kwargs)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model

def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)

    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24

    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone, **kwargs):

    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, **kwargs)
    elif backbone.startswith('hrnetv2'):
        model = _segm_hrnet(arch_type, backbone, num_classes, pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model


# Deeplab v3
def deeplabv3_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False): # no pretrained backbone yet
    return _load_model('deeplabv3', 'hrnetv2_48', output_stride, num_classes, pretrained_backbone=pretrained_backbone)

def deeplabv3_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True):
    return _load_model('deeplabv3', 'hrnetv2_32', output_stride, num_classes, pretrained_backbone=pretrained_backbone)

def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=False)

def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


# Deeplab v3+
def deeplabv3plus_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False, **kwargs): # no pretrained backbone yet
    return _load_model('deeplabv3plus', 'hrnetv2_48', num_classes, output_stride, pretrained_backbone=pretrained_backbone, **kwargs)


def deeplabv3plus_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'hrnetv2_32', num_classes, output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet34(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    return _load_model('deeplabv3plus', 'resnet34', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, **kwargs)


def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, **kwargs)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, **kwargs)


def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
