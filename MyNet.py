import torch
import torch.nn as nn
from deform_conv_v2 import *
import torch.nn.functional as F
import torch.nn as nn
import math

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


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResCut(nn.Module):
    expansion = 1
    short_cuts = []

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResCut, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        ResCut.short_cuts.append(residual)
        # dense shortcut
        # TODO(lds)改成weight normalized，不好再做修改
        for short_cut in ResCut.short_cuts:
            out += short_cut

        # out += residual

        out = self.relu(out)

        return x


class MyNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # TODO(lds) AAM
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.adptpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(576, num_classes)
        # 对卷积和与BN层初始化，论文中也提到过
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        # init
        downsample = None
        block.short_cuts = []

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("conv size", x.size())
        x = self.maxpool(x)
        # print("conv size", x.size())
        x = self.layer1(x)
        gapool1 = self.adptpool(x)
        # x = self.aam1(x)
        # print("layer1", x.size())
        x = self.layer2(x)
        # gapool2 = self.adptpool(x)
        # print("layer2", x.size())
        x = self.aam2(x)
        x = self.layer3(x)
        x = self.aam3(x)
        # gapool3 = self.adptpool(x)
        # print("layer3", x.size())
        x = self.layer4(x)
        gapool4 = self.adptpool(x)
        # print("layer4", x.size())
        # print('gappool size', gapool1.size())
        gapool1 = gapool1.view(gapool1.size(0), -1)
        # print("gap1 view size", gapool1.size())
        # gapool2 = gapool2.view(gapool2.size(0), -1)
        # gapool3 = gapool3.view(gapool3.size(0), -1)
        gapool4 = gapool4.view(gapool4.size(0), -1)
        x = torch.cat([gapool1,gapool4], dim=1)
        # print()
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.adptpool(x)
        # print("avgpool", x.size())
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

def mynet(pretrained=False, modelpath='./models',**kwargs):
    model = MyNet(ResCut, [6, 6, 6, 6], **kwargs)
    return model








