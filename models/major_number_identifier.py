import torch.nn as nn

class MajorNumberIdentifier(nn.Module):

  def __init__(
      self, 
      n_numbers: int,
      hidden_dim: int,
      conv_layers: int = 0,
      fc_layers:int = 1,
      cnn_kernel_size:int = 1,
      activation: str = "Identity"
  ):
      super().__init__()
      self.num_embedding = nn.Embedding(n_numbers, hidden_dim)

      self.cnn = nn.Identity()
      if conv_layers > 0:
        self.cnn = nn.Sequential(
          *[
            nn.Sequential(
              nn.Conv1d(hidden_dim, hidden_dim, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
              getattr(nn, activation)()
            ) for _ in range(conv_layers)
          ]
        )

      self.pool = nn.AdaptiveMaxPool1d(output_size=1)

      self.fc = nn.Sequential(
        *[
          nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            getattr(nn, activation)()
          ) for _ in range(fc_layers)
        ],
        nn.Linear(hidden_dim, 1)
      )

  def forward(self, x):
      x = self.num_embedding(x).permute(0, 2, 1)
      x = self.cnn(x)
      x = self.pool(x).squeeze(-1)
      x = self.fc(x).squeeze(-1)
      return x
