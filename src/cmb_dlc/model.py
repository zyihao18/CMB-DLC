"""Retained Brain-CMD CompactUNet25D architecture, copied without evaluation code.

Source: code/inference/models.py at local revision
 dc1b552ea5238eb93170e8c182ee515167b025b9 (see docs/SOURCE_AUDIT.md).
Original class implementation preserved; original repository supplies no LICENSE.
Repository publication awaits source provenance/access checks.
"""
import torch
from torch import nn

class CompactUNet25D(nn.Module):
    """
    Compact U-Net for 2.5D lesion segmentation.
    Processes multiple slices but outputs segmentation for all slices.
    Supports multiple input channels.
    """

    def __init__(self, in_channels: int = 1, num_slices: int = 5, out_channels: int = 1):
        super().__init__()

        self.num_slices = num_slices

        # For 2.5D, we treat slices as additional channels
        # Input: (B, C, D, H, W) -> reshape to (B, C*D, H, W)
        effective_in_channels = in_channels * num_slices

        # Encoder
        self.enc1 = self._conv_block(effective_in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)

        # Bottleneck
        self.bottleneck = self._conv_block(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32)

        # Output - predict for all slices
        self.out_conv = nn.Conv2d(32, out_channels * num_slices, kernel_size=1)

        # Pooling
        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Convolutional block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, 1, D, H, W)
        """
        B, C, D, H, W = x.shape

        # Reshape: (B, C, D, H, W) -> (B, C*D, H, W)
        x = x.reshape(B, C * D, H, W)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output: (B, D, H, W)
        out = self.out_conv(d1)
        out = out.reshape(B, D, H, W)

        # Add channel dimension: (B, 1, D, H, W)
        out = out.unsqueeze(1)

        return torch.sigmoid(out)
