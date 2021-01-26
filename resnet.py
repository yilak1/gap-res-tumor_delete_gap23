import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from triplet_attention import *
from MyNet import *
__all__ = ['ResNet', 'resnet34', 'resnet101', 'resnet50']

model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.triplet_attention = TripletAttention(planes, 16, no_spatial=True)
        self.se = SEModule(planes * self.expansion)
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.triplet_attention(out)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.adptpool = nn.AdaptiveAvgPool2d((1, 1))
        self.aam1 = AAM(64)
        self.aam2 = AAM(128)
        self.aam3 = AAM(256)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc1 = nn.Linear(576, num_classes)
        #self.fc2 = nn.Linear(600, num_classes)
        #self.fc = nn.Linear(51200, num_classes)
        # 对卷积和与BN层初始化，论文中也提到过
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # 这里是为了结局两个残差块之间可能维度不匹配无法直接相加的问题，
    # 相同类型的残差块只需要改变第一个输入的维数就好，后面的输入维数都等于输出维数
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 扩维度使channel数一致
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
        #print("conv size", x.size())
        x = self.maxpool(x)
        #print("conv size", x.size())
        x = self.layer1(x)
        gapool1 = self.adptpool(x)
        x = self.aam1(x)
        # print("layer1", x.size())
        x = self.layer2(x)
        #gapool2 = self.adptpool(x)
        # print("layer2", x.size())
        x = self.aam2(x)
        x = self.layer3(x)
        x = self.aam3(x)
        #gapool3 = self.adptpool(x)
        # print("layer3", x.size())
        x = self.layer4(x)
        gapool4 = self.adptpool(x)
        # print("layer4", x.size())
        # print('gappool size', gapool1.size())
        gapool1 = gapool1.view(gapool1.size(0), -1)
        # print("gap1 view size", gapool1.size())
        #gapool2 = gapool2.view(gapool2.size(0), -1)
        #gapool3 = gapool3.view(gapool3.size(0), -1)
        gapool4 = gapool4.view(gapool4.size(0), -1)
        x = torch.cat([gapool1,gapool4], dim=1)
        # print()
        x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.adptpool(x)
        #print("avgpool", x.size())
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x





def resnet34(pretrained=False, modelpath='./models',**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir=modelpath))
    return model

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet101(pretrained=False, modelpath='./models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir=modelpath))
    return model