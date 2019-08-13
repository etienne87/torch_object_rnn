import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights

        if not bilinear:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        if hasattr(self, 'up'):
            x1 = self.up(x1)
        else:
            x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base, n_layers):
        super(UNet, self).__init__()
        self.inc = inconv(in_channels, base)

        self.downs = []
        self.ups = []

        self.channels = [base]

        for i in range(n_layers):
            channels = min(base * 2 ** (n_layers-1), self.channels[-1] * 2)
            self.channels.append(channels)
            self.downs.append( down(self.channels[-2], self.channels[-1]) )

        self.channels.pop()

        for i in range(n_layers):
            channels = self.channels.pop()
            in_ch = channels * 2
            out_ch = max(base, channels /2)
            self.ups.append(up(in_ch, out_ch) )

        self.outc = outconv(base, out_channels)


    def forward(self, x):
        self.encoded = [self.inc(x)]

        for down_layer in self.downs:
            self.encoded.append(down_layer(self.encoded[-1]))

        x = self.encoded.pop()

        for up_layer in self.ups:
            x = up_layer(x, self.encoded.pop())

        x = self.outc(x)
        return x

if __name__ == '__main__':
    unet = UNet(3, 2, 64, 4)

    x = torch.rand(1, 3, 256, 256)

    y = unet(x)

    print(y.shape)