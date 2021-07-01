import torch
from torch.utils.data import Dataset

import numpy as np

from collections import Counter

class BinaryImageDataset(Dataset):

  def __init__(
      self, 
      image_size :int, 
      major_class_prob: float,
      num_images: int,
      dataset_length: int,
      p: float = 0.5
  ):
    super().__init__()
    self.image_size = image_size
    self.major_class_prob = major_class_prob
    self.num_images = num_images
    self.dataset_length = dataset_length
    self.p = p

    self.all_images = torch.vstack(
        [torch.floor(torch.arange(2 ** image_size) / (2 ** i)) % 2 for i in range(image_size)]
    ).permute(1, 0)

    self.image_count = self.all_images.shape[0]
    self.probs = (1 - self.major_class_prob) / (self.image_count-1) * np.ones((self.image_count, ))
    maj_idx = np.random.randint(0, self.image_count)
    self.probs[maj_idx] = self.major_class_prob 

  def __len__(self):
    return self.dataset_length

  def __getitem__(self, idx):
    indices = np.random.choice(np.arange(self.image_count), size=self.num_images, p=self.probs)

    c_ = Counter(indices)
    most_common_count = c_.most_common()[0][1]

    label = 1 if most_common_count >= self.num_images * 0.5 else 0      

    return self.all_images[indices].reshape((1, -1)), label
    