import torch
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics

from ..models import MajorNumberIdentifier

class _MajorityIdentifierBase_PL(pl.LightningModule):

  def __init__(
      self, 
      **kwargs
  ):
      super().__init__()
      raise NotImplementedError("This is an abstract class.")

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    images, labels = batch
    logits = self(images)
    pred_labels = (logits >= 0).int()

    loss = F.binary_cross_entropy_with_logits(logits, labels.float())

    return {'loss' : loss, 'preds': pred_labels, 'labels': labels}

  def validation_step(self, batch, batch_idx):
    self.training_step(batch, batch_idx)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
    return {"optimizer" : optimizer, "scheduler" : scheduler, "monitor" : "train/loss"}

  def training_epoch_end(self, outputs):
    preds = torch.cat([o['preds'] for o in outputs])
    labels = torch.cat([o['labels'] for o in outputs])
    acc = torchmetrics.functional.accuracy(preds, labels)
    self.log('train/accuracy', acc)

  def validation_epoch_end(self, outputs):
    preds = torch.cat([o['preds'] for o in outputs])
    labels = torch.cat([o['labels'] for o in outputs])
    acc = torchmetrics.functional.accuracy(preds, labels)
    self.log('val/accuracy', acc)

class MajorNumberIdentifier_PL(_MajorityIdentifierBase_PL):

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
      self.model = MajorNumberIdentifier(n_numbers, hidden_dim, conv_layers, fc_layers, cnn_kernel_size, activation)
