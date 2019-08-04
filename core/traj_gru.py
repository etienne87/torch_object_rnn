"""
this script is stolen from
https://github.com/Hzzone/Precipitation-Nowcasting
"""
import torch
from torch import nn
import torch.nn.functional as F



class Activation(torch.nn.Module):

    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super(Activation, self).__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError


# input: B, C, H, W
# flow: [B, 2, H, W]
def warp(input, flow):
    B, C, H, W = input.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(input.device)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(input.device)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(input, vgrid)
    return output


class BaseConvRNN(nn.Module):
    def __init__(self, num_filter, b_h_w,
                 h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                 i2h_kernel=(3, 3), i2h_stride=(1, 1),
                 i2h_pad=(1, 1), i2h_dilate=(1, 1),
                 act_type=torch.tanh,
                 prefix='BaseConvRNN'):
        super(BaseConvRNN, self).__init__()
        self._prefix = prefix
        self._num_filter = num_filter
        self._h2h_kernel = h2h_kernel
        assert (self._h2h_kernel[0] % 2 == 1) and (self._h2h_kernel[1] % 2 == 1), \
            "Only support odd number, get h2h_kernel= %s" % str(h2h_kernel)
        self._h2h_pad = (h2h_dilate[0] * (h2h_kernel[0] - 1) // 2,
                         h2h_dilate[1] * (h2h_kernel[1] - 1) // 2)
        self._h2h_dilate = h2h_dilate
        self._i2h_kernel = i2h_kernel
        self._i2h_stride = i2h_stride
        self._i2h_pad = i2h_pad
        self._i2h_dilate = i2h_dilate
        self._act_type = act_type
        assert len(b_h_w) == 3
        i2h_dilate_ksize_h = 1 + (self._i2h_kernel[0] - 1) * self._i2h_dilate[0]
        i2h_dilate_ksize_w = 1 + (self._i2h_kernel[1] - 1) * self._i2h_dilate[1]
        self._batch_size, self._height, self._width = b_h_w
        self._state_height = (self._height + 2 * self._i2h_pad[0] - i2h_dilate_ksize_h)\
                             // self._i2h_stride[0] + 1
        self._state_width = (self._width + 2 * self._i2h_pad[1] - i2h_dilate_ksize_w) \
                             // self._i2h_stride[1] + 1
        self._curr_states = None
        self._counter = 0
        self.states = None

    def reset(self):
        self._curr_states = None

    # stateful warping
    def forward(self, inputs):
        out, self._curr_states = self.forward_stateless(inputs, self._curr_states)
        return out

    def forward_stateless(self, inputs, states=None):
        return Exception('Not Implemented')



class TrajGRU(BaseConvRNN):
    # b_h_w: input feature map size
    def __init__(self, input_channel, num_filter, b_h_w, zoneout=0.0, L=5,
                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                 h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                 act_type=Activation('leaky', negative_slope=0.2, inplace=True)):
        super(TrajGRU, self).__init__(num_filter=num_filter,
                                      b_h_w=b_h_w,
                                      h2h_kernel=h2h_kernel,
                                      h2h_dilate=h2h_dilate,
                                      i2h_kernel=i2h_kernel,
                                      i2h_pad=i2h_pad,
                                      i2h_stride=i2h_stride,
                                      act_type=act_type,
                                      prefix='TrajGRU')
        self._L = L
        self._zoneout = zoneout

        # wxz, wxr, wxh
        # reset_gate, update_gate, new_mem
        self.i2h = nn.Conv2d(in_channels=input_channel,
                            out_channels=self._num_filter*3,
                            kernel_size=self._i2h_kernel,
                            stride=self._i2h_stride,
                            padding=self._i2h_pad,
                            dilation=self._i2h_dilate)

        # inputs to flow
        self.i2f_conv1 = nn.Conv2d(in_channels=input_channel,
                                out_channels=32,
                                kernel_size=(5, 5),
                                stride=1,
                                padding=(2, 2),
                                dilation=(1, 1))

        # hidden to flow
        self.h2f_conv1 = nn.Conv2d(in_channels=self._num_filter,
                                   out_channels=32,
                                   kernel_size=(5, 5),
                                   stride=1,
                                   padding=(2, 2),
                                   dilation=(1, 1))

        # generate flow
        self.flows_conv = nn.Conv2d(in_channels=32,
                                   out_channels=self._L * 2,
                                   kernel_size=(5, 5),
                                   stride=1,
                                   padding=(2, 2))


        # hh, hz, hr
        self.ret = nn.Conv2d(in_channels=self._num_filter*self._L,
                                   out_channels=self._num_filter*3,
                                   kernel_size=(1, 1),
                                   stride=1)



    # inputs: B*C*H*W
    def _flow_generator(self, inputs, states):
        if inputs is not None:
            i2f_conv1 = self.i2f_conv1(inputs)
        else:
            i2f_conv1 = None
        h2f_conv1 = self.h2f_conv1(states)
        f_conv1 = i2f_conv1 + h2f_conv1 if i2f_conv1 is not None else h2f_conv1
        f_conv1 = self._act_type(f_conv1)

        flows = self.flows_conv(f_conv1)
        flows = torch.split(flows, 2, dim=1)
        return flows

    # inputs: S*B*C*H*W
    def forward_stateless(self, inputs=None, states=None):
        if states is None:
            states = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float).to(inputs.device)
        if inputs is not None:
            S, B, C, H, W = inputs.size()
            i2h = self.i2h(torch.reshape(inputs, (-1, C, H, W)))
            i2h = torch.reshape(i2h, (S, B, i2h.size(1), i2h.size(2), i2h.size(3)))
            i2h_slice = torch.split(i2h, self._num_filter, dim=2)

        else:
            i2h_slice = None

        seq_len = inputs.size(0)
        prev_h = states
        outputs = []
        for i in range(seq_len):
            if inputs is not None:
                flows = self._flow_generator(inputs[i, ...], prev_h)
            else:
                flows = self._flow_generator(None, prev_h)
            warped_data = []
            for j in range(len(flows)):
                flow = flows[j]
                warped_data.append(warp(prev_h, -flow))
            warped_data = torch.cat(warped_data, dim=1)
            h2h = self.ret(warped_data)
            h2h_slice = torch.split(h2h, self._num_filter, dim=1)
            if i2h_slice is not None:
                reset_gate = torch.sigmoid(i2h_slice[0][i, ...] + h2h_slice[0])
                update_gate = torch.sigmoid(i2h_slice[1][i, ...] + h2h_slice[1])
                new_mem = self._act_type(i2h_slice[2][i, ...] + reset_gate * h2h_slice[2])
            else:
                reset_gate = torch.sigmoid(h2h_slice[0])
                update_gate = torch.sigmoid(h2h_slice[1])
                new_mem = self._act_type(reset_gate * h2h_slice[2])
            next_h = update_gate * prev_h + (1 - update_gate) * new_mem
            if self._zoneout > 0.0:
                next_h = F.dropout2d(prev_h, p=self._zoneout)
            outputs.append(next_h)
            prev_h = next_h

        # return torch.cat(outputs), next_h
        return torch.stack(outputs), next_h


if __name__ == '__main__':
    tbins, batchsize, channels, height, width = 4, 2, 32, 64, 64
    out_channels = 64

    x = torch.zeros((tbins, batchsize, channels, height, width), dtype=torch.float32)

    rnn = TrajGRU(channels, out_channels, (batchsize, height, width), zoneout=0.1)

    y = rnn(x)

    print(y.shape)