import torch
import torch.nn as nn
Base = 32

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)
class Conv_11(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class DoubleConv_s2(nn.Module):
    def __init__(self, in_channels, out_channels, stride, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)
class Down(nn.Module):
    def __init__(self, in_channels, out_channels,stride):
        super().__init__()
        self.conv_s2 = nn.Sequential(
            DoubleConv_s2(in_channels, out_channels,stride)
        )
    def forward(self, x):
        return self.conv_s2(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels,stride,increase_channels = True):
        super().__init__()
        if increase_channels:
            if (in_channels%out_channels) ==0:
                mid_channel = in_channels
            else:
                mid_channel = in_channels + Base*3
        else:
            mid_channel = in_channels + in_channels//2
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=stride, stride=stride)
        self.conv = DoubleConv(mid_channel , out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class SingleLinear(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.linear_1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout)
        )
    def forward(self, x):
        return self.linear_1(x)


class OutConv(nn.Module):  
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
    
class CompSegNet(nn.Module):   ##   CompSegNet --> Compositional Segmentaiton Network
    def __init__(self, n_channels = 1):
        super(CompSegNet, self).__init__()
        self.n_channels = n_channels
        self.inc = DoubleConv(n_channels, Base)
        self.down1 = Down(Base, 2*Base,2)
        self.down2 = Down(2*Base, 4*Base,2)
        self.down3 = Down(4*Base, 8*Base,2)
        self.down4 = Down(8*Base, 10*Base,2)
        self.down5 = Down(10*Base, 10*Base,2)
        self.up0 = Up(10*Base, 10*Base,2, False)
        self.up1 = Up(10*Base, 8*Base,2)
        self.up2 = Up(8*Base, 4*Base,2)
        self.up3 = Up(4*Base, 2*Base,2)
        self.up4 = Up(2*Base, Base,2)
        self.up0_ = Up(10*Base, 10*Base,2,False)
        self.up1_ = Up(10*Base, 8*Base,2)
        self.up2_ = Up(8*Base, 4*Base,2)
        self.up3_ = Up(4*Base, 2*Base,2)
        self.up4_ = Up(2*Base, Base,2)
        self.outc = OutConv(Base,2)
        self.outc4 = OutConv(Base,4)
        self.dropout6E = nn.Dropout3d(p=0.30) 
    def forward(self, x):
        
        ## Image Encoder ###
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) 
        x6 = self.down5(x5) 
        x6 = self.dropout6E(x6)
         ## Super-Segmentation Decoder ##
        z1 = self.up0(x6, x5)
        z2 = self.up1(z1, x4)
        z3 = self.up2(z2, x3)
        z4 = self.up3(z3, x2)
        z5 = self.up4(z4, x1)
        logits2 = self.outc(z5)
        ## Sub-Segmentation Decoder ##
        y1 = self.up0_(x6, z1)
        y2 = self.up1_(y1, z2)
        y3 = self.up2_(y2, z3)
        y4 = self.up3_(y3, z4)
        y5 = self.up4_(y4, z5)
        logits4 = self.outc4(y5)
        return logits2,logits4
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.SegNetwork = CompSegNet()
    def forward(self, x):
        logits2, logits4 = self.SegNetwork(x)
        return logits2, logits4
    
    
from torchsummary import summary
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model()
    model.to(device=DEVICE, dtype=torch.float)
    summary(model, [(1, 256, 256)])
    
    image = torch.randn(10, 1, 256, 256).to(device=DEVICE, dtype=torch.float)
    output = model(image)
    
