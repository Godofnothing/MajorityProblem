import torch.nn as nn

from ..modules import Conv1d

class HierarchicalModel(nn.Module):

  def __init__(
      self, 
      depth, 
      hidden_dim: int,
      image_size: int,
      branching_number = 2,
      activation: str = None,
      batchnorm: bool = False
  ):
      super().__init__()

      self.in_conv = Conv1d(
          in_channels=1,
          out_channels=hidden_dim,
          kernel_size=image_size,
          stride=image_size,
          activation=activation,
          batchnorm=batchnorm   
      )

      self.net = nn.Sequential(*[
          Conv1d(
              in_channels=hidden_dim,
              out_channels=hidden_dim,
              kernel_size=branching_number,
              stride=branching_number,
              activation=activation,
              batchnorm=batchnorm 
          ) for i in range(depth)
      ])

      self.head = nn.Linear(hidden_dim, 1)

  def forward(self, x):
      x = self.in_conv(x)
      x = self.net(x).squeeze(-1)
      x = self.head(x).squeeze(-1)
      return x
      