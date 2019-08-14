import torch
import torch.nn as nn
import torch.nn.functional as F

def time_to_batch(x):
    t, n, c, h, w = x.size()
    x = x.view(n * t, c, h, w)
    return x, n


def batch_to_time(x, n=32):
    nt, c, h, w = x.size()
    time = int(nt / n)
    x = x.view(time, n, c, h, w)
    return x


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

        self.downs = nn.ModuleList(self.downs)
        self.ups = nn.ModuleList(self.ups)


    def forward(self, x):
        self.encoded = [self.inc(x)]

        for down_layer in self.downs:
            self.encoded.append(down_layer(self.encoded[-1]))

        x = self.encoded.pop()

        for up_layer in self.ups:
            x = up_layer(x, self.encoded.pop())

        x = self.outc(x)
        return x


class UNetLSTM(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(UNetLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.x2h = UNet(in_channels, 4 * self.hidden_dim, 16, 3)
        self.h2h = UNet(self.hidden_dim, 4 * self.hidden_dim, 16, 3)
        self.act = torch.tanh
        self.prev_hidden = None
        self.reset()
        self.alpha = 1.0

    def detach_hidden(self):
        if isinstance(self.prev_hidden, list):
            for i in range(len(self.prev_hidden)):
                if self.prev_hidden[i] is not None:
                    self.prev_hidden[i] = self.prev_hidden[i].detach()
        elif self.prev_hidden is not None:
            self.prev_hidden = self.prev_hidden.detach()

    def reset(self, mask=None):
        if mask is None or self.prev_hidden is None:
            self.prev_hidden = None
        else:
            if isinstance(self.prev_hidden, list):
                for item in self.prev_hidden:
                    if item is not None:
                        item *= mask
            elif self.prev_hidden is not None:
                self.prev_hidden *= mask

    def forward(self, x):
        self.detach_hidden()
        xseq = x.unbind(0)
        result = []
        ht = None
        xt = None

        # First treat sequence
        for t, xt in enumerate(xseq):
            ht = self.forward_lstm(xt)
            result.append(ht[None])

        result = torch.cat(result, dim=0)
        return result

    def forward_lstm(self, xt):
        if self.prev_hidden is None:
            prev_h, prev_c = None, None
        else:
            prev_h, prev_c = self.prev_hidden

        if self.prev_hidden is None:
            tmp = self.x2h(xt)
        else:
            tmp = self.x2h(xt) + self.h2h(prev_h)

        cc_i, cc_f, cc_o, cc_g = torch.split(tmp, self.hidden_dim, dim=1)  # tmp.chunk(4, 1) #
        f = torch.sigmoid(cc_f)
        i = torch.sigmoid(cc_i)
        o = torch.sigmoid(cc_o)
        g = self.act(cc_g)

        if self.prev_hidden:
            c = f * prev_c + i * g
        else:
            c = i * g

        h = o * self.act(c)
        self.prev_hidden = [h, c]
        return h


if __name__ == '__main__':
    unet = UNet(3, 2, 16, 2)
    x = torch.rand(1, 3, 64, 64)
    y = unet(x)
    print(y.shape)

    unet_rnn = UNetLSTM(3, 16)

    seq = torch.rand(3, 2, 3, 64, 64)

    y = unet_rnn(seq)
    print(y.shape)