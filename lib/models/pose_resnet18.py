import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

BN_MOMENTUM = 0.1

# initialize paramaters
def init_conv_bn(module):
    with torch.no_grad():

        # for all modules
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

# Make sequential of ConvTranspose2d layers
class DeconvLayers(nn.Module):
    def __init__(self, num_layers, num_filters, num_kernels, input_channels):
        super().__init__()
        assert num_layers == len(num_filters) == len(num_kernels)

        self.blocks = nn.ModuleList()
        current_c = input_channels

        for i in range(num_layers):
            out_c = num_filters[i]
            k = num_kernels[i]

            block = nn.Sequential(
                nn.ConvTranspose2d(current_c, out_c, kernel_size=k, stride=2,
                                   padding=1, output_padding=0, bias=False),
                nn.BatchNorm2d(out_c, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )
            self.blocks.append(block)
            current_c = out_c

        self.out_channels = current_c

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class PoseResNet18(nn.Module):

    def __init__(self, num_joints):
        super().__init__()
        num_joints = num_joints
        kernel_final = 3
        pad = 1 if kernel_final == 3 else 0

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # since we don't need avgpool, fc, we only get front parts from backbone
        # (B, 3, H, W)
        self.conv1   = backbone.conv1  # 7x7, stride=2, padding=3 → (B, 64, H/2,  W/2)
        self.bn1     = backbone.bn1                             # → (B, 64, H/2,  W/2)
        self.relu    = backbone.relu                            # → (B, 64, H/2,  W/2)
        self.maxpool = backbone.maxpool # 3x3, stride=2, pad=1    → (B, 64, H/4,  W/4)
        self.layer1  = backbone.layer1 # stride=1                 → (B, 64,  H/4,  W/4)
        self.layer2  = backbone.layer2 # stride=2                 → (B, 128, H/8,  W/8)
        self.layer3  = backbone.layer3 # stride=2                 → (B, 256, H/16, W/16)
        self.layer4  = backbone.layer4 # stride=2                 → (B, 512, H/32, W/32)

        # output channels from layer4 is 512
        input_channels = 512

        # (B, 512, H/32, W/32)
        #  -> Deconv #1: (B, 256, H/16, W/16)
        #  -> Deconv #2: (B, 256, H/8,  W/8)
        #  -> Deconv #3: (B, 256, H/4,  W/4)
        self.deconv = DeconvLayers(
            num_layers=3,
            num_filters=[256, 256, 256],
            num_kernels=[4, 4, 4],
            input_channels=input_channels
        )
        out_channels = self.deconv.out_channels

        # output: (B, num_joints, H/4,  W/4)
        self.final = nn.Conv2d(out_channels, num_joints, kernel_size=kernel_final, stride=1, padding=pad)

        init_conv_bn(self.deconv)
        init_conv_bn(self.final)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv(x)
        out = self.final(x)
        return out