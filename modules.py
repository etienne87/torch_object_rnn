import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
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
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
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


'''
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
'''
class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super(AddCoords,self).__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super(CoordConv,self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


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