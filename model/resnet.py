from torch.nn import Module
import torch.nn as nn
import torch
import torchvision


def conv3x(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)


def conv1x(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)


class BasicBlock(Module):
    expansion: int = 1

    def __init__(self, input_channels, output_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x(input_channels, output_channels, stride)
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x(output_channels, output_channels)
        self.bn2 = nn.BatchNorm1d(output_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if self.downsample is not None:
            residual = self.downsample(x)

        y += residual
        y = self.relu(y)

        return y


class ResNet(Module):
    def __init__(self, input_channels, block, layers, num_classes=20, task='multilabel'):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(input_channels, self.in_channels, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.type = task
        if self.type == 'multilabel':
            self.act = nn.Sigmoid()
        elif self.type == 'multiclass':
            self.act = nn.Softmax(dim=-1)

    def _make_layer(self, block, channels, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                conv1x(self.in_channels, channels * block.expansion, stride),
                nn.BatchNorm1d(channels * block.expansion)
            )
        layers = [block(self.in_channels, channels, stride, downsample)]
        self.in_channels = channels * block.expansion
        for i in range(1, block_num):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x:(batch_size, chanel, length)
        # x: (batch_size, 1, 784)
        x = self.conv1(x)
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
        x = self.act(x)

        return x


def resnet1d34(input_channels=12, num_classes=20, task='multilabel'):
    return ResNet(input_channels, BasicBlock, [3, 4, 6, 3], num_classes, task=task)


def resnet50(pretrained=None, num_classes=4):
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights_backbone=pretrained, num_classes=num_classes)
    model.backbone["conv1"] = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model
