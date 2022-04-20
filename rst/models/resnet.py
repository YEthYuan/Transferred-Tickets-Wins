import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from utils.builder import get_builder
from args import args

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

# BasicBlock {{{
class BasicBlock(nn.Module):
    M = 2
    expansion = 1

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(BasicBlock, self).__init__()
        if base_width / 64 > 1:
            raise ValueError("Base width >64 does not work for BasicBlock")

        self.conv1 = builder.conv3x3(inplanes, planes, stride)
        self.bn1 = builder.batchnorm(planes)
        self.relu = builder.activation()
        self.conv2 = builder.conv3x3(planes, planes)
        self.bn2 = builder.batchnorm(planes, last_bn=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# BasicBlock }}}

# Bottleneck {{{
class Bottleneck(nn.Module):
    M = 3
    expansion = 4

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * base_width / 64)
        self.conv1 = builder.conv1x1(inplanes, width)
        self.bn1 = builder.batchnorm(width)
        self.conv2 = builder.conv3x3(width, width, stride=stride)
        self.bn2 = builder.batchnorm(width)
        self.conv3 = builder.conv1x1(width, planes * self.expansion)
        self.bn3 = builder.batchnorm(planes * self.expansion, last_bn=True)
        self.relu = builder.activation()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out


# Bottleneck }}}

# ResNet {{{
class ResNet(nn.Module):
    def __init__(self, builder, block, layers, num_classes=1000, base_width=64, normalize=None):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.base_width = base_width
        if self.base_width // 64 > 1:
            print(f"==> Using {self.base_width // 64}x wide model")

        if args.first_layer_dense:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = builder.conv7x7(3, 64, stride=2, first_layer=True)

        self.bn1 = builder.batchnorm(64)
        self.relu = builder.activation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(builder, block, 64, layers[0])
        self.layer2 = self._make_layer(builder, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(builder, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(builder, block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.normalize = normalize

        # if args.last_layer_dense:
        #     self.fc = nn.Conv2d(512 * block.expansion, args.num_classes, 1)
        # else:
        #     self.fc = builder.conv1x1(512 * block.expansion, num_classes)

    def _make_layer(self, builder, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = builder.conv1x1(
                self.inplanes, planes * block.expansion, stride=stride
            )
            dbn = builder.batchnorm(planes * block.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = []
        layers.append(block(builder, self.inplanes, planes, stride, downsample, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)

        x = self.conv1(x)

        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# ResNet }}}
def ResNet18(pretrained=False, num_classes=1000, norm=None):
    model = ResNet(get_builder(), BasicBlock, [2, 2, 2, 2], num_classes=num_classes, normalize=norm)
    if pretrained:
        print(" => Use pytorch pretrained model")
        state_dict = load_state_dict_from_url(model_urls['resnet18'],
                                              progress=True)
        model.load_state_dict(state_dict, strict=False)
    return model

def ResNet34(pretrained=False, num_classes=1000, norm=None):
    return ResNet(get_builder(), BasicBlock, [3, 4, 6, 3], num_classes=num_classes, normalize=norm)

def ResNet50(pretrained=False, num_classes=1000, norm=None):
    model = ResNet(get_builder(), Bottleneck, [3, 4, 6, 3], num_classes=num_classes, normalize=norm)
    if pretrained:
        print(" => Use pytorch pretrained model")
        state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=True)
        model.load_state_dict(state_dict, strict=False)
    return model


def ResNet101(pretrained=False, num_classes=1000, norm=None):
    return ResNet(get_builder(), Bottleneck, [3, 4, 23, 3], num_classes=num_classes, normalize=norm)


def WideResNet50_2(pretrained=False, num_classes=1000, norm=None):
    return ResNet(
        get_builder(), Bottleneck, [3, 4, 6, 3], num_classes=num_classes, base_width=64 * 2, normalize=norm
    )


def WideResNet101_2(pretrained=False, num_classes=1000, norm=None):
    return ResNet(
        get_builder(), Bottleneck, [3, 4, 23, 3], num_classes=num_classes, base_width=64 * 2, normalize=norm
    )

