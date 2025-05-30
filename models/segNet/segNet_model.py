from .segNet_layers import *

class segNet(nn.Module):
    
    def __init__(self, n_channels):
        super(segNet, self).__init__()
        self.n_channels = n_channels

        # Encoder
        self.in_conv = DoubleConv(n_channels, 64, stride=1)
        self.down1 = Down(64, 128, stride=1)
        self.down2 = Down(128, 256, stride=1)
        self.down3 = Down(256, 512, stride=1)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 512)

        # Decoder
        self.up2 = Up(512, 256, stride=1)
        self.up3 = Up(256, 128, stride=1)
        self.up4 = Up(128, 64, stride=1)

        # Output
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        
        
    def forward(self, x):
        x = self.in_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.bottleneck(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.out_conv(x)
        return x
