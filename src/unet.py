import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from torch.autograd import Variable

import vgg as VGG
import grid as GR
import numpy as np

def predict(net, in_size, out_size, X):
    assert len(X.shape) == 3, "It's only one image!"
    assert (in_size - out_size) % 2 == 0, "parity violation"

    size1 = X.shape[1]
    size2 = X.shape[2]

    grid1 = GR.eval_grid(size1, out_size)
    grid2 = GR.eval_grid(size2, out_size)

    padding = (in_size - out_size) // 2
    X_expanded = GR.reflect_padding(X, padding)

    Y = np.empty((1, size1, size2))

    for u in grid1:
        for v in grid2:
            x = X_expanded[None, :, u:u+in_size, v:v+in_size]
            x = Variable(FloatTensor(x))

            if torch.cuda.is_available():
                y = net(x.cuda()).cpu()
            else:
                y = net(x)

            y = y.data.numpy()[0]
            Y[:, u:u + out_size, v:v + out_size] = y

    return Y

class Unet(nn.Module):
    def __init__(self, vgg_pretrained=False, n_classes=2):
        super(Unet, self).__init__()

        self.bn = nn.BatchNorm2d(3)

        vgg = VGG.create_vgg16_bn(pretrained=vgg_pretrained)
        self.encoder = Encoder(vgg)

        self.decoder = Decoder()
        self.conv1x1 = nn.Conv2d(64, n_classes, 1)

        initialize_weights(self.bn)
        initialize_weights(self.conv1x1)
        initialize_weights(self.decoder)

        if not vgg_pretrained:
            initialize_weights(self.encoder)

    def forward(self, x):
        encs = self.encoder(self.bn(x))
        dec = self.decoder(*encs)
        logits = self.conv1x1(dec)
        probs = F.softmax(logits, dim=1)
        return probs

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
    def __init__(self, conv_block_type='B'):
        super(Decoder, self).__init__()

        conv_block = conv_factory(conv_block_type)
        self.bottleneck = conv_block(512, 1024, 512)
        self.merge1 = MergeBlock(conv_block, 4, 512)
        self.merge2 = MergeBlock(conv_block, 18, 256)
        self.merge3 = MergeBlock(conv_block, 46, 128)
        self.merge4 = MergeBlock(conv_block, 100, 64, out_channels=64)

    def forward(self, x, y32, y22, y12, y5):
        x = self.bottleneck(x)
        x = self.merge1(x, y32)
        x = self.merge2(x, y22)
        x = self.merge3(x, y12)
        x = self.merge4(x, y5)
        return x

class MergeBlock(nn.Module):
    def __init__(self, conv_block, crop_size, in_channels, out_channels=None):
        assert in_channels % 2 == 0, "in_channels must be even"

        if out_channels is None:
            out_channels = in_channels // 2

        super(MergeBlock, self).__init__()

        self.crop_size = crop_size
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = conv_block(2 * in_channels, in_channels, out_channels)

    def forward(self, x, y):
        x = self.upsample(x)
        y = crop(y, self.crop_size)
        z = torch.cat([x, y], dim=1)
        return self.conv(z)

    
def crop(x, s):
    a = x.data.shape[2]
    b = x.data.shape[3]
    return x[:, :, s:a-s, s:b-s]

def conv_factory(c_type='B'):
    if c_type == 'A':
        return conv_block_A
    elif c_type == 'B':
        return conv_block_B
    else:
        raise "Pass A or B"

def conv_block_A(in_channels, middle_channels, out_channels):
    x1 = nn.Conv2d(in_channels, middle_channels, 3)
    x2 = nn.BatchNorm2d(middle_channels)
    x3 = nn.ReLU()
    x4 = nn.Conv2d(middle_channels, out_channels, 3)
    x5 = nn.BatchNorm2d(out_channels)
    x6 = nn.ReLU()
    return nn.Sequential(x1,x2,x3,x4,x5,x6)

def conv_block_B(in_channels, middle_channels, out_channels):
    x1 = nn.BatchNorm2d(in_channels)
    x2 = nn.Conv2d(in_channels, middle_channels, 3)
    x3 = nn.ReLU()
    x4 = nn.Conv2d(middle_channels, out_channels, 3)
    x5 = nn.ReLU()
    return nn.Sequential(x1,x2,x3,x4,x5)

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # m.weight.data.normal_(0, math.sqrt(2. / n))
            torch.nn.init.xavier_uniform(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
