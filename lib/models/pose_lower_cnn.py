import torch
import torch.nn as nn

BN_MOMENTUM = 0.1

def init_conv_bn(module):
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class DilatedResidualBlock(nn.Module):

    def __init__(self, channels, dilation=1):
        super().__init__()
        padding = dilation

        # First 3x3 dilated conv
        self.conv1 = nn.Conv2d(
            # input channels
            channels,
            # output channels
            channels,
            kernel_size=3,
            stride=1,
            padding=padding,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Second 3x3 dilated conv
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=padding,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)

    def forward(self, x):
        # first conv → bn → relu
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        # second conv → bn
        y = self.conv2(y)
        y = self.bn2(y)

        # # add residual
        y = y + x
        y = self.relu(y)
        return y


class LowerBodyPoseNet(nn.Module):
    """
    Input : (B, 3, 256, 192)
    Output: (B, num_joints, 64, 48)
    """
    def __init__(self, num_joints=6, in_channels=3):
        super().__init__()
        self.num_joints = num_joints

        # first conv block
        self.conv1 = nn.Conv2d(
            in_channels,
            32,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.bn1 = nn.BatchNorm2d(
            32,
            momentum=BN_MOMENTUM
        )
        self.relu1 = nn.ReLU()

        # second conv block
        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=5,
            stride=2,
            padding=2
        )
        self.bn2 = nn.BatchNorm2d(
            64,
            momentum=BN_MOMENTUM
        )
        self.relu2 = nn.ReLU()

        # dilated residual blocks
        self.block1 = DilatedResidualBlock(
            channels=64,
            dilation=1
        )
        self.block2 = DilatedResidualBlock(
            channels=64,
            dilation=2
        )
        self.block3 = DilatedResidualBlock(
            channels=64,
            dilation=4
        )
        self.block4 = DilatedResidualBlock(
            channels=64,
            dilation=1
        )

        self.head_conv1 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.head_bn1 = nn.BatchNorm2d(
            num_features=64,
            momentum=BN_MOMENTUM
        )
        self.head_relu1 = nn.ReLU()
        self.head_conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=num_joints,
            kernel_size=1,
            stride=1,
            padding=0
        )

        init_conv_bn(self)

    def forward(self, x):
        # first conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # second conv
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # dilated residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.head_conv1(x)
        x = self.head_bn1(x)
        x = self.head_relu1(x)

        heatmaps = self.head_conv2(x)
        return heatmaps
