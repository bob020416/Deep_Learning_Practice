import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class ResNet34UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet34UNet, self).__init__()

        self.in_channels = 64

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        # build basic block based on resnet 

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # downsample bottleneck 

        # Concate and Upsampling and Decoder
        self.upconv4 = nn.ConvTranspose2d(768, 768, kernel_size=2, stride=2)  #trans conv 
        self.decoder4 = self._block(512 + 256, 32)  

        self.upconv3 = nn.ConvTranspose2d(288, 288, kernel_size=2, stride=2) 
        self.decoder3 = self._block(256 + 32, 32) 

        self.upconv2 = nn.ConvTranspose2d(160, 160, kernel_size=2, stride=2)  
        self.decoder2 = self._block(128 + 32, 32)  

        self.upconv1 = nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2) 
        self.decoder1 = self._block(64 + 32, 32)  

        # Final layer
        self.upconv0 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)  # 32 to 32
        self.conv_last = nn.Conv2d(32, out_channels, kernel_size=1)  # 32 to out_channels


# Helper Function ------------------------------------------------------------------
    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
# -----------------------------------------------------------------------------------

    def forward(self, x):
        e1 = self.input_layer(x)
        e2 = self.layer1(e1)
        e3 = self.layer2(e2)
        e4 = self.layer3(e3)
        e5 = self.layer4(e4)

        b = self.bottleneck(e5)

        d4 = self.upconv4(torch.cat((e5, b), dim=1))
        d4 = self.decoder4(d4)

        d3 = self.upconv3(torch.cat((d4, e4), dim=1))
        d3 = self.decoder3(d3)

        d2 = self.upconv2(torch.cat((d3, e3), dim=1))
        d2 = self.decoder2(d2)

        d1 = self.upconv1(torch.cat((d2, e2), dim=1))
        d1 = self.decoder1(d1)

        out = self.upconv0(d1)

        out = self.conv_last(out)

        return out
