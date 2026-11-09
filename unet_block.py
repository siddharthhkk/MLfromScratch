import torch.nn as nn
import torch
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride= 2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU()
        )
    def forward(self, x, skip):
        up = self.up(x)
        concat = torch.cat([up, skip], dim = 1)
        return self.conv(concat)

x = torch.randn(64, 64, 14, 14)
skip = torch.randn(64, 32, 28, 28)

up_block = UNetUpBlock(in_channels=64, out_channels=32)
out = up_block(x, skip)
print(out.shape)