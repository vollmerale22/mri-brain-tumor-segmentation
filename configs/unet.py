"""
U-Net Architecture for 3D Medical Image Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DoubleConv(nn.Module):
    """
    Double Convolution block: Conv3D -> BatchNorm -> ReLU -> Conv3D -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle padding if sizes don't match
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric medical image segmentation.
    
    Args:
        in_channels: Number of input channels (MRI modalities)
        out_channels: Number of output channels (segmentation classes)
        features: List of feature map sizes for each level
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        features: List[int] = [32, 64, 128, 256]
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        
        # Encoder (downsampling)
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        # Bottleneck
        self.bottleneck = Down(features[3], features[3] * 2)
        
        # Decoder (upsampling)
        self.up1 = Up(features[3] * 2, features[3])
        self.up2 = Up(features[3], features[2])
        self.up3 = Up(features[2], features[1])
        self.up4 = Up(features[1], features[0])
        
        # Output layer
        self.outc = nn.Conv3d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck
        x5 = self.bottleneck(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing UNet3D...")
    
    model = UNet3D(in_channels=4, out_channels=4, features=[32, 64, 128, 256])
    
    # Create dummy input (batch_size=2, channels=4, depth=64, height=64, width=64)
    x = torch.randn(2, 4, 64, 64, 64)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {model.count_parameters():,}")
    
    # Expected output shape: (2, 4, 64, 64, 64)
    assert output.shape == (2, 4, 64, 64, 64), "Output shape mismatch!"
    print("✓ Model test passed!")