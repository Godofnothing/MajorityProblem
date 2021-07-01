import torch
import torch.nn as nn

from typing import Union

class Conv1d(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 0,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        activation: str = 'Identity',
        batchnorm: bool = False,
        padding_mode = 'zeros'
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias, padding_mode
        )

        self.activation = getattr(nn, activation)
        
        self.bn = nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.activation(self.bn(self.conv(x)))