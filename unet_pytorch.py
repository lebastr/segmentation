import torch
import torch.nn as nn
from torch.utils import model_zoo
import torch.nn.functional as F
import math

class Unet(nn.Module):
    def __init__(self, vgg_pretrained=False):
        super(Unet, self).__init__()

        self.bn = nn.BatchNorm2d(3)

        vgg = create_vgg16_bn(pretrained=vgg_pretrained)
        self.encoder = Encoder(vgg)

        self.decoder = Decoder()
        self.conv1x1 = nn.Conv2d(64, 1, 1)

        initialize_weights(self.bn)
        initialize_weights(self.conv1x1)
        initialize_weights(self.decoder)

        if not vgg_pretrained:
            initialize_weights(self.encoder)

    def forward(self, x):
        outs = self.encoder(self.bn(x))
        y = self.conv1x1(self.decoder(*outs))
        return F.sigmoid(y)

class Encoder(nn.Module):
    def __init__(self, vgg):
        super(Encoder, self).__init__()
        for idx, layer in enumerate(list(vgg.features)[:34]):
            self.add_module(str(idx), layer)

    def forward(self, x):
        outputs = []
        idx = [5, 12, 22, 32, 33]

        i = 0
        for j, layer in enumerate(self.children()):
            x = layer(x)
            if j == idx[i]:
                outputs.append(x)
                i += 1

        outputs.reverse()
        return outputs

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.bottleneck = conv_bn_relu(512, 1024, 512)
        self.merge1 = MergeBlock(4, 512)
        self.merge2 = MergeBlock(18, 256)
        self.merge3 = MergeBlock(46, 128)
        self.merge4 = MergeBlock(100, 64, out_channels=64)

    def forward(self, x, y32, y22, y12, y5):
        x = self.bottleneck(x)
        x = self.merge1(x, y32)
        x = self.merge2(x, y22)
        x = self.merge3(x, y12)
        x = self.merge4(x, y5)
        return x

class MergeBlock(nn.Module):
    def __init__(self, crop_size, in_channels, out_channels=None):
        assert in_channels % 2 == 0, "in_channels must be even"

        if out_channels is None:
            out_channels = in_channels // 2

        super(MergeBlock, self).__init__()

        self.crop_size = crop_size
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = conv_bn_relu(2 * in_channels, in_channels, out_channels)

    def forward(self, x, y):
        x = self.upsample(x)
        y = crop(y, self.crop_size)
        z = torch.cat([x, y], dim=1)
        return self.conv(z)

def crop(x, s):
    a = x.data.shape[2]
    b = x.data.shape[3]
    return x[:, :, s:a-s, s:b-s]

def conv_bn_relu(in_channels, middle_channels, out_channels):
    x1 = nn.Conv2d(in_channels, middle_channels, 3)
    x2 = nn.BatchNorm2d(middle_channels)
    x3 = nn.ReLU()
    x4 = nn.Conv2d(middle_channels, out_channels, 3)
    x5 = nn.BatchNorm2d(out_channels)
    x6 = nn.ReLU()

    return nn.Sequential(x1,x2,x3,x4,x5,x6)

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



def create_vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model