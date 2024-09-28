from torch.nn import Module
import torch.nn as nn
import torch
import torchvision


def vgg_block1d(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv1d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg_block2d(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)


class VGG1d11(Module):
    def __init__(self, input_channels, num_classes=20):
        super(VGG1d11, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        # conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        #
        # conv_blks = []
        # in_channels = self.input_channels
        # # 卷积层部分
        # for (num_convs, out_channels) in conv_arch:
        #     conv_blks.append(vgg_block1d(num_convs, in_channels, out_channels))
        #     in_channels = out_channels
        #
        # self.conv_layers = nn.Sequential(*conv_blks)
        #
        # self.linear_layers = nn.Sequential(
        #     nn.Flatten(),
        #     # 全连接层部分
        #     nn.Linear(conv_arch[-1][-1] * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        #     nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        #     nn.Linear(4096, num_classes))

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.AdaptiveAvgPool1d(7)
        )

        self.linear_layers = torch.nn.Sequential(
            nn.Linear(512 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x


class VGG11(Module):
    def __init__(self, n_channels, n_classes):
        super(VGG11, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=self.n_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x


def vgg1d11(input_channels=12, num_classes=20):
    return VGG1d11(input_channels=input_channels, num_classes=num_classes)


def vgg2d11(input_channels=1, num_classes=4):
    return VGG11(n_channels=input_channels, n_classes=num_classes)
