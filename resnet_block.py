import torch

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels)
        )
        if stride == 1 and in_channels == out_channels:
            self.shortcut = torch.nn.Sequential()
        elif stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels),)

    def forward(self, x):
        self.out = self.conv(x)
        self.out = self.out + self.shortcut(x)
        return torch.nn.functional.relu(self.out)


x = torch.randn(64, 32, 14, 14)
block1 = ResidualBlock(32, 32)
print(block1(x).shape)
block2 = ResidualBlock(32, 64, stride=2)
print(block2(x).shape)

