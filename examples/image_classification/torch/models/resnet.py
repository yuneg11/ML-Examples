from functools import partial

from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


def resnet(
    layers,
    block,
    num_classes: int,
    width_per_group: int = 64,
    image_shape=None,
    **kwargs,
):
    return ResNet(block, layers, num_classes, width_per_group=width_per_group, **kwargs)


ResNet18  = partial(resnet, layers=[2,  2,  2, 2], block=BasicBlock)
ResNet34  = partial(resnet, layers=[3,  4,  6, 3], block=BasicBlock)
ResNet50  = partial(resnet, layers=[3,  4,  6, 3], block=Bottleneck)
ResNet101 = partial(resnet, layers=[3,  4, 23, 3], block=Bottleneck)
ResNet152 = partial(resnet, layers=[3,  8, 36, 3], block=Bottleneck)
ResNet200 = partial(resnet, layers=[3, 24, 36, 3], block=Bottleneck)
