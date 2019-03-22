import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
import torch


def time_to_batch(x):
    t, n, c, h, w = x.size()
    x = x.view(n * t, c, h, w)
    return x, n


def batch_to_time(x, n=32):
    nt, c, h, w = x.size()
    time = int(nt / n)
    x = x.view(time, n, c, h, w)
    return x

class ClampMod(Function):
    r""" clamp (gradient 1)
     """

    @staticmethod
    def forward(ctx, x, minV, maxV):
        bx = torch.clamp(x, minV, maxV)
        return bx

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output
        return grad_x, None, None

clampMod = ClampMod.apply


def mod_sigmoid2(x, alpha=0.2):
    o = F.sigmoid(x) * (1. + 2. * alpha) - alpha
    o = clampMod(o, 0, 1)
    return o


# scaled and clamped tanh forward, gradient of scaled tanh backward
def mod_tanh2(x, alpha=0.2):
    o = F.tanh(x) * (1. + alpha)
    o = clampMod(o, -1, 1)
    return o


# hard sigmoid forward, gradient 1 from -0.5-alpha to 0.5+alpha
def mod_sigmoid(x, alpha=0.1):
    o = torch.clamp(x + 0.5, 0 - alpha, 1 + alpha)
    o = clampMod(o, 0, 1)
    return o


# hard tanh forward, gradient 1 from -1-alpha to 1+alpha
def mod_tanh(x, alpha=0.1):
    o = torch.clamp(x, -1 - alpha, 1 + alpha)
    o = clampMod(o, -1, 1)
    return o


def get_nonlinearity(mode):
    if mode == "hard":
        sigmoid = mod_sigmoid
        tanh = mod_tanh
    elif mode == "soft":
        sigmoid = mod_sigmoid2
        tanh = mod_tanh2
    elif mode == "original":
        sigmoid = F.sigmoid
        tanh = F.tanh
    return sigmoid, tanh


class ConvLSTMCell(nn.Module):
    r"""ConvLSTMCell module, applies sequential part of LSTM.
    """
    def __init__(self, hidden_dim, kernel_size, bias, nonlin=F.leaky_relu):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim

        self.conv_h2h = nn.Conv2d(in_channels=self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=kernel_size,
                                  padding=1,
                                  bias=bias)

        self.reset()
        self.nonlin = nonlin

        self.sigmoid, self.tanh = get_nonlinearity("original")

    def forward(self, xi):
        xiseq = xi.split(1, 0) #t,n,c,h,w


        if self.prev_h is not None:
            self.prev_h = self.prev_h.detach()
            self.prev_c = self.prev_c.detach()

        result = []
        for t, xt in enumerate(xiseq):
            xt = xt.squeeze(0)

            if self.prev_h is not None:
                tmp = self.conv_h2h(self.prev_h) + xt
            else:
                tmp = xt

            cc_i, cc_f, cc_o, cc_g = torch.split(tmp, self.hidden_dim, dim=1)
            i = self.sigmoid(cc_i)
            f = self.sigmoid(cc_f)
            o = self.sigmoid(cc_o)
            g = self.tanh(cc_g)
            if self.prev_c is None:
                c = i * g
            else:
                c = f * self.prev_c + i * g
            h = o * self.nonlin(c)
            result.append(h.unsqueeze(0))
            self.prev_h = h
            self.prev_c = c
        res = torch.cat(result, dim=0)
        return res

    def reset(self):
        self.prev_h, self.prev_c = None, None


class ConvLSTM(nn.Module):
    r"""ConvLSTM module. computes input-to-hidden in parallel.
    """
    def __init__(self, nInputPlane, nOutputPlane, kernel_size, stride, padding, dilation=1):
        super(ConvLSTM, self).__init__()

        self.cin = nInputPlane
        self.cout = nOutputPlane
        self.conv1 = nn.Conv2d(nInputPlane, 4 * nOutputPlane, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=padding,
                               bias=True)

        self.bn1 = nn.BatchNorm2d(nInputPlane)

        self.timepool = ConvLSTMCell(nOutputPlane, 3, True)

    def forward(self, x):
        x, n = time_to_batch(x)
        bnx = self.bn1(x)
        y = self.conv1(bnx)
        y = batch_to_time(y, n)
        h = self.timepool(y)
        return h


class CoordConv(nn.Module):
    """
    coord conv implementation https://eng.uber.com/coordconv
    """

    def __init__(self, in_size, out_channels, img_rows, img_cols, kernel=1,
                 **kwargs):
        super(CoordConv, self).__init__()
        grid_h = (torch.arange(img_rows)[:, None] * torch.ones(img_rows, img_cols) - 0.5 * img_rows) * 2 / img_rows
        grid_w = (torch.arange(img_cols)[None, :] * torch.ones(img_rows, img_cols) - img_cols * 0.5) * 2 / img_cols
        self.grid = nn.Parameter(torch.cat((grid_h[None, :, :],
                                            grid_w[None, :, :]), 0), False)
        self.conv = nn.Conv2d(in_size + 2, out_channels, kernel, **kwargs)

    def forward(self, inp):
        x = torch.cat(
            (inp, self.grid[None, ...].repeat(inp.shape[0], 1, 1, 1)), 1)
        return self.conv(x)


class Conv2d(nn.Module):

    def __init__(self, cin, cout, kernel_size, stride, padding, dilation=1, addcoords=False):
        super(Conv2d, self).__init__()
        self.cin = cin
        self.cout = cout

        if addcoords:
            self.cin += 2
            self.conv1 = CoordConv(cin, cout, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                   padding=padding,
                                   bias=True)
        else:
            self.conv1 = nn.Conv2d(cin, cout, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                   padding=padding,
                                   bias=True)

        self.bn1 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        return x


class SeparableConv2D(nn.Module):
    """Convolve spatially channels independently & then pointwise together"""

    def __init__(self, cin, cout, kernel, depth_multiplier=1, stride=1,
                 padding=1):
        super(SeparableConv2D, self).__init__()

        self.dw = nn.Conv2d(cin, cin * depth_multiplier, kernel, padding=padding,
                            stride=stride, groups=cin)

        self.bn1 = nn.GroupNorm(cin, cin * depth_multiplier)
        self.pw = nn.Conv2d(cin * depth_multiplier, cout, 1,
                            stride=1)
        self.bn2 = nn.BatchNorm2d(cout)

        self.output_channels = cout

    def forward(self, x):
        x1 = self.dw(x)

        x1 = self.bn1(x1)
        out = F.relu(self.pw(x1))
        out = self.bn2(out)
        return out


class SepConvLSTM(nn.Module):
    """
    convlstm with a depthwise separable convolution first
    """

    def __init__(self, cin, cout, kernel_size, padding, stride=1):
        super(SepConvLSTM, self).__init__()

        self.conv1 = SeparableConv2D(cin, 4 * cout, kernel_size, depth_multiplier=2, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(cin)

        self.timepool = ConvLSTMCell(cout, 3, True)

    def forward(self, x):
        x, n = time_to_batch(x)
        bnx = self.bn1(x)
        y = self.conv1(bnx)
        y = batch_to_time(y, n)
        h = self.timepool(y)
        return h

    def reset(self):
        """
        reset memory of the lstm cells
        """
        self.timepool.reset()



class ResNetBlock_concat(nn.Module):
    """Residual Block using concatenate"""

    def __init__(self, cin, cout=-1):
        super(ResNetBlock_concat, self).__init__()

        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        return torch.cat((conv2, x), 1)


class ResNetBlock(nn.Module):
    """Transition block"""

    def __init__(self, cin):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(cin, cin, 3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv2 = nn.Conv2d(cin, cin, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(cin)

    def forward(self, x):
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        return conv2 + x


class UpSampleConv(nn.Module):
    """interpolate with nearest neighbors then convolves"""

    def __init__(self, cin, cout, kernel, scale_factor=2, non_linearity=lambda s: s, **kwargs):
        super(UpSampleConv, self).__init__()
        self.non_linearity = non_linearity
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(cin, cout, kernel, **kwargs)
        self.bn1 = nn.BatchNorm2d(cout)

    def forward(self, x):
        conv1 = self.conv1(F.interpolate(x, scale_factor=self.scale_factor))
        return self.non_linearity(self.bn1(conv1))
