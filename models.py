from torch.nn import init, Tanh, Dropout, Parameter, ConvTranspose2d, AdaptiveAvgPool2d, Softmax, InstanceNorm2d, Flatten, PReLU, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm1d, BatchNorm2d, Linear, Upsample, Sigmoid, LeakyReLU, MultiheadAttention
from torch.nn.functional import interpolate, adaptive_avg_pool2d
from torchvision import models
import torchvision.transforms as transforms
import torch

class Encoder(Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.layers = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            Interpolate(size=(8,8)),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

# NOTE: CORRECT ORDER
class Classifier(Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.layers = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            Linear(in_features=128*8*8, out_features=256),
            ReLU(inplace=True),
            Linear(in_features=256, out_features=128),
            ReLU(inplace=True),
            Linear(in_features=128, out_features=5),
            Softmax(dim=1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

# NOTE: WRONG ORDER
"""class Classifier(Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.layers = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            MaxPool2d(kernel_size=2, stride=2),
            BatchNorm2d(32),
            Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            BatchNorm2d(64),
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            BatchNorm2d(128),
            Flatten(),
            Linear(in_features=128*8*8, out_features=256),
            ReLU(inplace=True),
            Linear(in_features=256, out_features=128),
            ReLU(inplace=True),
            Linear(in_features=128, out_features=5),
            Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x"""

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
    def __init__(self, size=None, scale_factor=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.size is not None:
            x = interpolate(x, size=self.size, mode='bilinear', align_corners=False)
        elif self.scale_factor is not None:
            x = interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        else:
            raise ValueError("Either 'size' or 'scale_factor' must be specified.")
        return x

class Generator(Module):
    def __init__(self, noise_dim=105):
        super(Generator, self).__init__()

        self.linear = Linear(noise_dim, 4 * 4 * 1024)
        self.layers = Sequential(
            LeakyReLU(0.2, inplace=True),
            Interpolate(scale_factor=2),
            Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512),
            LeakyReLU(0.2, inplace=True),
            Interpolate(scale_factor=2),
            Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
            LeakyReLU(0.2, inplace=True),
            Interpolate(scale_factor=2),
            Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128),
            LeakyReLU(0.2, inplace=True),
            SelfAttention(in_channels=128),
            Interpolate(scale_factor=2),
            Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            LeakyReLU(0.2, inplace=True),
            SelfAttention(in_channels=64),
            Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            Tanh(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 1024, 4, 4)
        x = self.layers(x)
        return x

class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            LeakyReLU(0.2, inplace=True),
            #Dropout(0.5),
            Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(128),
            LeakyReLU(0.2, inplace=True),
            #Dropout(0.5),
            Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(256),
            LeakyReLU(0.2, inplace=True),
            #Dropout(0.5),
            Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(512),
            LeakyReLU(0.2, inplace=True),
            #Dropout(0.5),
            Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(1024),
            LeakyReLU(0.2, inplace=True),
            #Dropout(0.5),
            Flatten(),
            Linear(4*4*1024, 1),
        )
        self.sigmoid = Sigmoid()
        #self.noise = GaussianNoise()

    def forward(self, x):
        #x = self.noise(x)
        logits = self.layers(x)
        output = self.sigmoid(logits)
        return (logits, output)

class GaussianNoise(Module):
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * 0.1  # Adjust the noise level as needed
            x = x + noise
        return x

class VGG19(Module):
    def __init__(self, pretrained=True, require_grad=False):
        super(VGG19, self).__init__()
        vgg_features = models.vgg19(pretrained=pretrained).features
        selected_layers = [vgg_features[i] for i in [0, 5, 10, 19, 25]]
        self.layers = Sequential(*selected_layers)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        for parameter in self.parameters():
            parameter.requires_grad = require_grad

    def forward(self, x):
        x = transpose_image(x, range_min=0, range_max=1)
        x = self.normalize(x)
        x = self.layers(x)
        x = transpose_image(x, range_min=0, range_max=1)
        return x

# NOTE: PRE-TRAINED VGG19 CLASSIFIER (SELECTED LAYERS)
"""class VGG19(Module):
    def __init__(self, pretrained=True, require_grad=False, num_classes=5):
        super(VGG19, self).__init__()
        vgg_features = models.vgg19(pretrained=pretrained).features

        selected_layers = [vgg_features[i] for i in [0, 5, 10, 19, 28]]
        self.layers = Sequential(*selected_layers)
        self.avgpool = adaptive_avg_pool2d
        self.classifier = Sequential(
            Linear(512 * 7 * 7, 4096),
            ReLU(True),
            Linear(4096, 4096),
            ReLU(True),
            Linear(4096, num_classes)
        )

        if not require_grad:
            for parameter in self.parameters():
                parameter.requires_grad = True

    def forward(self, x):
        features = self.layers(x)
        pooled_features = self.avgpool(features, (7, 7))
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        output = self.classifier(pooled_features)
        return output"""
