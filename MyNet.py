import torch
import torch.nn as nn
from deform_conv_v2 import *
import torch.nn.functional as F
import torch.nn as nn


def conv(inc,outc, kernel_size=3, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    return nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=kernel_size, stride=stride)


class AAM(nn.Module):

    def __init__(self, inc, de=4):
        super(AAM, self).__init__()
        self.deform_conv1 = DeformConv2d(inc, inc // de, kernel_size=3, stride=1)
        self.deform_conv2 = DeformConv2d(inc, inc // de, kernel_size=3, stride=1)
        self.conv1 = conv(inc=inc, outc=inc, kernel_size=1, stride=1)
        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, x):
        # [batch_size, channel, h, w]
        # print(x.size())
        f, offset1 = self.deform_conv1(x)
        # print("f", f.size())
        g, offset2 = self.deform_conv2(x)
        # print("g", g.size())
        h = self.conv1(x)
        # print("h", h.size())
        # [batch_size, channel//de, N, N] N: h * w
        s = torch.matmul(torch.transpose(f.view(f.size(0), f.size(1), -1), 1, 2),
                         g.view(g.size(0), g.size(1), -1)) #[batch_size, ]
        # [batch_size, N, N]
        # print("s",s.size())
        beta_a = self.soft_max(s)
        # print("beta", beta_a.size())
        # [batch_size, C, N] * [batch_size, N, N]
        o = torch.matmul(h.view(h.size(0), h.size(1), -1), beta_a)
        # print("o", o.size())
        gamma = nn.Parameter(torch.zeros(1))
        # [batch_size, channel, h, w]
        o = o.view(x.size())
        atta = gamma * o
        x = x + atta
        # print("x1", x.size())
        return offset1, offset2, x












