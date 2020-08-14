import torch
from torchvision.models.resnet import Bottleneck, ResNet, model_urls
from torchvision.models.utils import load_state_dict_from_url


class ResNetWithRep(ResNet):
    """
    Wrapper class of torchvision.models.resnet.ResNet.
    This class allow to return representation.
    """
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetWithRep, self).__init__(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual,
                                            groups=groups, width_per_group=width_per_group, replace_stride_with_dilation=replace_stride_with_dilation,
                                            norm_layer=norm_layer)

    def _forward_impl(self, x, return_rep):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        rep = torch.flatten(x, 1)
        logit = self.fc(rep)

        if return_rep:
            return logit, rep
        else:
            return logit

    def forward(self, x, return_rep=False):
        return self._forward_impl(x, return_rep=return_rep)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetWithRep(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
