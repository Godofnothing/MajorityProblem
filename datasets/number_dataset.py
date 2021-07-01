import torch
from torch.utils.data import Dataset

import numpy as np

from collections import Counter

class NumberDataset(Dataset):

    def __init__(
      self, 
      major_num_prob: float,
      n_numbers: int,
      numbers_per_sample: int, 
      dataset_length: int
    ):
      super().__init__()
      self.major_num_prob = major_num_prob
      self.n_numbers = n_numbers
      self.numbers_per_sample = numbers_per_sample
      self.dataset_length = dataset_length

      self.numbers = torch.arange(n_numbers)
      self.probs = (1 - major_num_prob) / (n_numbers - 1) * np.ones(shape=(n_numbers,))
      maj_idx = np.random.randint(0, n_numbers)
      self.probs[maj_idx] = major_num_prob

    def __len__(self):
      return self.dataset_length

    def __getitem__(self, idx):
      indices = np.random.choice(np.arange(self.n_numbers), size=self.numbers_per_sample, p=self.probs)

      tmp_counter = Counter(indices)
      most_common_count = tmp_counter.most_common()[0][1]

      label = 1 if most_common_count >= self.numbers_per_sample * 0.5 else 0 

      return self.numbers[indices], label
      