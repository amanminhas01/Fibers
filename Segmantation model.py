import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

###############################################
# CoordConv
###############################################
class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        b, c, h, w = x.size()

        xx = torch.linspace(-1, 1, w).repeat(h, 1).to(x.device)
        yy = torch.linspace(-1, 1, h).repeat(w, 1).t().to(x.device)

        xx = xx.unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1)
        yy = yy.unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1)

        x = torch.cat([x, xx, yy], dim=1)

        return self.conv(x)


###############################################
# Soft Pooling (replacement for maxpool)
###############################################
class SoftPool2d(nn.Module):

    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        exp_x = torch.exp(x)

        num = F.avg_pool2d(x * exp_x, self.kernel_size, self.stride)
        den = F.avg_pool2d(exp_x, self.kernel_size, self.stride)

        return num / den


###############################################
# Conv Block
###############################################
class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


###############################################
# ASPP Module
###############################################
class ASPP(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)

        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=3, dilation=3)
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6)
        self.conv4 = nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12)
        self.conv5 = nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18)

        
        
       
        self.out = nn.Conv2d(out_ch * 5, out_ch, 1)

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)

        

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        return self.out(x)


###############################################
# Decoder Block
###############################################
class DecoderBlock(nn.Module):

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):

        x = self.up(x)

        x = torch.cat([x, skip], dim=1)

        x = self.conv(x)

        return x


###############################################
# Improved UNet with ResNet Encoder
###############################################
class ImprovedSegModel(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        resnet = models.resnet34(weights="IMAGENET1K_V1")

        ###################################
        # CoordConv first layer
        ###################################
        self.coord = CoordConv(3, 64)

        ###################################
        # Encoder
        ###################################
        self.encoder0 = resnet.relu

        self.pool = SoftPool2d()

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        ###################################
        # ASPP bottleneck
        ###################################
        self.aspp = ASPP(512, 256)
        ###################################
        # ASPP for skip connections
        ###################################
        
        self.aspp_skip4 = ASPP(256, 256)
        self.aspp_skip3 = ASPP(128, 128)
        self.aspp_skip2 = ASPP(64, 64)
        self.aspp_skip1 = ASPP(64, 64)
        ###################################
        # Decoder
        ###################################
        self.decoder4 = DecoderBlock(256, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        
        self.final_up = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.final_conv = nn.Conv2d(64 + 64, num_classes, 1)

    def forward(self, x):

        ###################################
        # CoordConv
        ###################################
        x0 = self.coord(x)

        x0 = self.encoder0(x0)

        ###################################
        # SoftPool
        ###################################
        x1 = self.pool(x0)

        ###################################
        # Encoder
        ###################################
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)
        
        ###################################
        # Bottleneck ASPP
        ###################################
        x5 = self.aspp(x5)
        
        ###################################
        # Apply ASPP to skips
        ###################################
        x4 = self.aspp_skip4(x4)
        x3 = self.aspp_skip3(x3)
        x2 = self.aspp_skip2(x2)
        x0 = self.aspp_skip1(x0)
        
        ###################################
        # Decoder
        ###################################
        d4 = self.decoder4(x5, x4)
        d3 = self.decoder3(d4, x3)
        d2 = self.decoder2(d3, x2)
        
        d0 = self.final_up(d2)
        
        out = torch.cat([d0, x0], dim=1)
        
        out = self.final_conv(out)

        return out
