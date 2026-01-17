from collections import OrderedDict
from typing import Any, Dict, List

import torch
import torch.nn as nn

from torch.nn.init import xavier_uniform_

from .activation_types import *


class ResidualBlock(nn.Module):
    '''
    Linear residual block.
    '''
    def __init__(self,
                 input_dim=2,
                 output_dim=1,
                 downsample=None):
        super().__init__()
        self.downsample = downsample

        self.fc = nn.Sequential(nn.Linear(input_dim, output_dim, bias=False),
                                 nn.ReLU(),
                                 nn.Linear(output_dim, output_dim, bias=False),
                                 nn.ReLU())
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.fc(x)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)

        return out


class ResNetLayer(nn.Module):
    '''
    Linear ResNet layer.
    '''
    def __init__(self,
                 input_dim=2,
                 output_dim=64,
                 n_blocks=2):
        super().__init__()

        self.downsample = None
        if input_dim != output_dim:
            self.downsample = nn.Linear(input_dim, output_dim, bias=False)

        self.blocks = nn.Sequential(
                ResidualBlock(input_dim=input_dim,
                      output_dim=output_dim,
                      downsample=self.downsample),
                *[ResidualBlock(input_dim=output_dim,
                        output_dim=output_dim,
                        downsample=None,
                        ) for _ in range(n_blocks-1)]
                )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNet(nn.Module):
    '''
    Linear ResNet.
    '''
    def __init__(self, configs: Dict[str, Any]):
        super(ResNet, self).__init__()

        self.configs = configs
        input_size = self.configs["input_size"]
        hidden_size = self.configs["hidden_units"][0]
        output_size = self.configs["output_size"]
        n_blocks = self.configs.get("resnet_blocks", 2)

        self.layer1 = ResNetLayer(input_dim=input_size,
                                  output_dim=hidden_size,
                                  n_blocks=n_blocks)
        self.layer2 = ResNetLayer(input_dim=hidden_size,
                                  output_dim=hidden_size,
                                  n_blocks=n_blocks)

        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc_out(out)
        return out