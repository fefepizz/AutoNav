from .uNet_layers import *
from torch.utils import checkpoint

class uNet(nn.Module):
    
    def __init__(self, n_channels):
        super(uNet, self).__init__()
        self.n_channels = n_channels

        # Encoder
        self.in_conv = DoubleConv(n_channels, 64, stride=1)
        self.down1 = Down(64, 128, stride=1)
        self.down2 = Down(128, 256, stride=1)
        self.down3 = Down(256, 512, stride=1)
        self.down4 = Down(512, 1024, stride=1)

        # Bottleneck
        self.bottleneck = DoubleConv(1024, 1024)

        # Decoder
        self.up1 = Up(1024, 512, stride=1)
        self.up2 = Up(512, 256, stride=1)
        self.up3 = Up(256, 128, stride=1)
        self.up4 = Up(128, 64, stride=1)

        # Output
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        
        
    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.bottleneck(x5)
        x7 = self.up1(x6, x4)
        x8 = self.up2(x7, x3)
        x9 = self.up3(x8, x2)
        x10 = self.up4(x9, x1)
        x11 = self.out_conv(x10)
        return x11
    
    def use_checkpoint(self):
        # for skip connections
        self.in_conv = checkpoint(self.in_conv)
        self.down1 = checkpoint(self.down1)
        self.down2 = checkpoint(self.down2)
        self.down3 = checkpoint(self.down3)
        self.down4 = checkpoint(self.down4)
        self.bottleneck = checkpoint(self.bottleneck)
        self.up1 = checkpoint(self.up1)
        self.up2 = checkpoint(self.up2)
        self.up3 = checkpoint(self.up3)
        self.up4 = checkpoint(self.up4)
        self.out_conv = checkpoint(self.out_conv)