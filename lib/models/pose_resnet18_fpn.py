import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

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

class DeconvLayers(nn.Module):
    """
    Simple stack of ConvTranspose2d + BN + ReLU blocks.
    """
    def __init__(self, num_layers, num_filters, num_kernels, input_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_c = input_channels

        for i in range(num_layers):
            out_c = num_filters[i]
            k = num_kernels[i]

            # Single deconv block: ConvTranspose2d → BatchNorm → ReLU
            block = nn.Sequential(
                nn.ConvTranspose2d(
                    current_c,
                    out_c,
                    kernel_size=k,
                    stride=2,
                    padding=1,
                    output_padding=0
                ),
                nn.BatchNorm2d(out_c, momentum=BN_MOMENTUM),
                nn.ReLU(),
            )
            self.blocks.append(block)

            # update input channel size for next layer
            current_c = out_c

        # Save final number of channels
        self.out_channels = current_c

    # Pass input into each deconv block
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class PoseResNet18FPN(nn.Module):
    """
    ResNet18 backbone + FPN + deconv head for pose estimation.
    """
    def __init__(self, num_joints):
        super().__init__()
        kernel_final = 3
        pad_final = 1 if kernel_final == 3 else 0

        # Back bone
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Input blocks
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Residual blocks
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Modify layer 4
        # After this, layer4 output is same as layer3

        # add dilation=2 for 3×3 convs
        # first block in layer4
        block0 = self.layer4[0]

        # conv1: stride1, dilation=2
        block0.conv1.stride = (1, 1)
        block0.conv1.dilation = (2, 2)
        block0.conv1.padding = (2, 2)

        # conv2: dilation=2
        block0.conv2.dilation = (2, 2)
        block0.conv2.padding = (2, 2)

        # downsample
        if block0.downsample is not None:
            block0.downsample[0].stride = (1, 1)

        # FPN layers
        # 1x1 convs to make all layer channels 256
        self.l2_proj = nn.Conv2d(128, 256, kernel_size=1)  # C=128
        self.l3_proj = nn.Conv2d(256, 256, kernel_size=1)  # C=256
        self.l4_proj = nn.Conv2d(512, 256, kernel_size=1)  # C=512

        self.l2_bn = nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        self.l3_bn = nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        self.l4_bn = nn.BatchNorm2d(256, momentum=BN_MOMENTUM)

        self.fpn_activation = nn.ReLU()

        # Deconv head

        # 1/8 -> 1/4
        self.deconv_head = DeconvLayers(
            num_layers=1,
            num_filters=[256],
            num_kernels=[4],
            input_channels=256,
        )
        deconv_out_channels = self.deconv_head.out_channels

        # Final conv for heatmaps(1/4 resolution)
        self.heatmap_head = nn.Conv2d(
            deconv_out_channels,
            num_joints,
            kernel_size=kernel_final,
            stride=1,
            padding=pad_final,
        )

        # Initialize only the newly added
        init_conv_bn(self.deconv_head)
        init_conv_bn(self.heatmap_head)

    def forward(self, x):
        """
        Return heatmaps: (B, num_joints, H/4, W/4)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # backbone features
        feat_4 = self.layer1(x)  # 1/4
        feat_8 = self.layer2(feat_4)  # 1/8
        feat_16 = self.layer3(feat_8)  # 1/16
        feat_16_dilated = self.layer4(feat_16)  # 1/16

        # FPN projection
        l2 = self.l2_bn(self.l2_proj(feat_8))
        l3 = self.l3_bn(self.l3_proj(feat_16))
        l4 = self.l4_bn(self.l4_proj(feat_16_dilated))

        # Upsample l3,l4
        target_hw = l2.shape[-2:]
        l3_up = F.interpolate(l3, size=target_hw, mode="bilinear", align_corners=False)
        l4_up = F.interpolate(l4, size=target_hw, mode="bilinear", align_corners=False)

        # merge features
        merged = l2 + l3_up + l4_up  # (256, H/8, W/8)
        merged = self.fpn_activation(merged)

        # deconv head
        result = self.deconv_head(merged)  # (256, H/4, W/4)

        # make heatmap
        heatmaps = self.heatmap_head(result)
        return heatmaps
