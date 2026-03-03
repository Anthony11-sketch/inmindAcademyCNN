import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _make_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class _ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = _make_block(in_channels, out_channels, stride)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x):
        return F.relu(self.block(x) + self.shortcut(x))


class ResNet20(nn.Module):
    """ResNet-20 for CIFAR-10 (32x32). Efficient and typically outperforms SimpleNet."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = nn.Sequential(
            _ResBlock(16, 16),
            _ResBlock(16, 16),
            _ResBlock(16, 16),
        )
        self.layer2 = nn.Sequential(
            _ResBlock(16, 32, stride=2),
            _ResBlock(32, 32),
            _ResBlock(32, 32),
        )
        self.layer3 = nn.Sequential(
            _ResBlock(32, 64, stride=2),
            _ResBlock(64, 64),
            _ResBlock(64, 64),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Simple CNN for CIFAR-10
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Use einops for clarity
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = rearrange(x, 'b c h w -> b (c h w)')  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x