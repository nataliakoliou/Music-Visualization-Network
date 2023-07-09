
from torch.nn import Tanh, Parameter, Softmax, Dropout, InstanceNorm2d, Flatten, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Linear, Sigmoid, LeakyReLU
from torch.nn.functional import interpolate
import torch

class Encoder(Module):
    def __init__(self, in_channels=3, feature_dim=4, base_channels=32, out_dim=10):
        super(Encoder, self).__init__()

        self.conv_blocks = Sequential(
            self._conv_block(in_channels, base_channels),
            self._conv_block(base_channels, base_channels * 2),
            self._conv_block(base_channels * 2, base_channels * 4),
            self._conv_block(base_channels * 4, base_channels * 8),
            self._conv_block(base_channels * 8, base_channels * 16),
            self._conv_block(base_channels * 16, base_channels * 32),
        )
        self.flat_layer = Flatten()
        self.linear_layer = Linear(base_channels * 32 * feature_dim * feature_dim // 2, out_dim)

    def _conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        layers = Sequential(
            Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
        )
        return layers

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.flat_layer(x)
        x = self.linear_layer(x)
        return x

class SelfAttention(Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        self.query = Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        attention_scores = torch.bmm(query, key)
        attention_scores = self.softmax(attention_scores)
        attention_output = torch.bmm(value, attention_scores.permute(0, 2, 1))
        attention_output = attention_output.view(batch_size, channels, height, width)
        x = self.gamma * attention_output + x
        return x
    
class Interpolate(Module):
    def __init__(self, scale_factor=None):
        super(Interpolate, self).__init__()
        
        self.scale_factor = scale_factor

    def forward(self, x):
        x = interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return x
    
class Generator(Module):
    def __init__(self, in_dim=100, feature_dim=4, base_channels=1024, out_channels=3):
        super(Generator, self).__init__()

        self.feature_dim = feature_dim
        self.base_channels = base_channels
        self.linear_block = Sequential(
            Linear(in_dim, feature_dim * feature_dim * base_channels),
            LeakyReLU(0.2, inplace=True)
        )
        self.deconv_blocks = Sequential(
            self._deconv_block(base_channels, base_channels // 2),
            self._deconv_block(base_channels // 2, base_channels // 4),
            self._deconv_block(base_channels // 4, base_channels // 8, attention=True),
            self._deconv_block(base_channels // 8, base_channels // 16, attention=True),
            self._deconv_block(base_channels // 16, out_channels, last=True)
        )
        self.activation_function = Tanh()

    def _deconv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, attention=False, last=False):
        if last:
            return Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            layers = [
                Interpolate(scale_factor=2),
                Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                InstanceNorm2d(out_channels),
                LeakyReLU(0.2, inplace=True)
            ]
            layers.append(SelfAttention(out_channels)) if attention else None
            return Sequential(*layers)

    def forward(self, x):
        x = self.linear_block(x)
        x = x.view(-1, self.base_channels, self.feature_dim, self.feature_dim)
        x = self.deconv_blocks(x)
        x = self.activation_function(x)
        return x

class Discriminator(Module):
    def __init__(self, in_channels=3, base_channels=64, feature_dim=4, out_dim=1):
        super(Discriminator, self).__init__()

        self.conv_blocks = Sequential(
            self._conv_block(in_channels, base_channels, normalize=False),
            self._conv_block(base_channels, base_channels * 2),
            self._conv_block(base_channels * 2, base_channels * 4),
            self._conv_block(base_channels * 4, base_channels * 8),
            self._conv_block(base_channels * 8, base_channels * 16, kernel_size=3, stride=1),
        )
        self.flat_layer = Flatten()
        self.linear_layer = Linear(base_channels * 16 * feature_dim * feature_dim, out_dim)
        self.activation_function = Sigmoid()

    def _conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalize=True, dropout=False):
        layers = [Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        layers.append(BatchNorm2d(out_channels)) if normalize else None
        layers.append(LeakyReLU(0.2, inplace=True))
        layers.append(Dropout(0.5)) if dropout else None
        return Sequential(*layers)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.flat_layer(x)
        logits = self.linear_layer(x)
        output = self.activation_function(logits)
        return (logits, output)
    
class Classifier(Module):
    def __init__(self, in_channels=3, feature_dim=8, base_channels=32, out_dim=5):
        super(Classifier, self).__init__()

        self.conv_blocks = Sequential(
            self._conv_block(in_channels, base_channels),
            self._conv_block(base_channels, base_channels * 2),
            self._conv_block(base_channels * 2, base_channels * 4),
        )
        self.flat_layer = Flatten()
        self.linear_blocks = Sequential(
            self._linear_block(base_channels * 4 * feature_dim * feature_dim, base_channels * 8),
            self._linear_block(base_channels * 8, base_channels * 4),
            self._linear_block(base_channels * 4, out_dim, last=True),
        )
        self.activation_function = Softmax(dim=1)

    def _conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        layers = Sequential(
            Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            BatchNorm2d(out_channels),
            LeakyReLU(0.2, inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        return layers

    def _linear_block(self, in_features, out_features, last=False):
        layers = [Linear(in_features, out_features)]
        layers.append(LeakyReLU(0.2, inplace=True)) if not last else None
        return Sequential(*layers)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.flat_layer(x)
        x = self.linear_blocks(x)
        x = self.activation_function(x)
        return x
