from torch.nn import Softmax, InstanceNorm2d, Flatten, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm1d, BatchNorm2d, Linear, Upsample, Sigmoid, LeakyReLU, MultiheadAttention
from torch.nn.functional import interpolate
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Optional, Tuple
import torch.nn.functional as F
import torch

"""class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        query_transposed = query.transpose(1, 2)
        score = torch.bmm(query, query_transposed ) / self.sqrt_dim
        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, query)
        return context, attn

class SelfAttention(Module):
    def __init__(self, in_channels, num_heads):
        super(SelfAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels % num_heads should be zero."
        self.d_head = int(in_channels / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.query_proj = nn.Linear(in_channels, self.d_head * num_heads)
        self.key_proj = nn.Linear(in_channels, self.d_head * num_heads)
        self.value_proj = nn.Linear(in_channels, self.d_head * num_heads)
    
    def forward(self, query: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_size = query.size(0)
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head) 
        key = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)     
        value = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head) 
        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)     
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1) 
        context, attn = self.scaled_dot_attn(query, key, value, mask)
        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)
        return context, attn"""

class SelfAttention(Module):
    def __init__(self, in_channels, num_heads):
        super(SelfAttention, self).__init__()
        self.multihead_attention = MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)
    
    def forward(self, x):
        batch_size, in_channels, width, length = x.size()
        x = x.permute(0, 2, 3, 1).reshape(batch_size, width * length, in_channels)
        output, _ = self.multihead_attention(x, x, x)  # (x, x, x) = (query, key, value)
        output = output.reshape(batch_size, width, length, in_channels).permute(0, 3, 1, 2)
        return output

class SelfAttention(Module):
    def __init__(self, in_channels, num_heads):
        super(SelfAttention, self).__init__()

class Encoder(Module):
    def __init__(self):
        super(Encoder, self).__init__()

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
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=1, kernel_size=1),
        )

    def forward(self, x):
        x = self.layers(x)
        upsampled_x = interpolate(x, size=(8, 8), mode='bilinear', align_corners=False)
        return upsampled_x
    
class Classifier(Module):
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
        return x

class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layers = Sequential(
            Upsample(scale_factor=2),
            Conv2d(in_channels=257, out_channels=256, kernel_size=3, stride=1, padding=1),
            InstanceNorm2d(256),
            LeakyReLU(0.2, inplace=True),
            Upsample(scale_factor=2),
            Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            InstanceNorm2d(128),
            LeakyReLU(0.2, inplace=True),
            #SelfAttention(in_channels=128, num_heads=4),
            #SelfAttention(in_channels=128, num_heads=4),
            Upsample(scale_factor=2),
            Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1),
            Sigmoid(),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x
    
class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(64),
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(128),
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(256),
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(256),
            LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
