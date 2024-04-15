from typing import List

import numpy as np
from sklearn import mixture
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


class MyGaussianMixture():
  def __init__(self):
    self.mixture = mixture.GaussianMixture(n_components=10)

  def fit(self, data: torch.Tensor):
    self.mixture.fit(data.numpy())

  def predict_one(self, point: torch.Tensor):
    return self.mixture.predict_proba(point.numpy()[np.newaxis, :])

  def predict_many(self, points: torch.Tensor):
    return self.mixture.predict_proba(points.numpy())


class MyLossFunction(nn.Module):
  def __init__(self, gmm: MyGaussianMixture):
    super(MyLossFunction, self).__init__()
    self.gmm = gmm

  def forward(self, embeddings, gts):
    gmm_predictions = self.gmm.predict_many(embeddings)
    # Ensure numerical stability
    epsilon = 1e-10
    # Clip embeddings to [1e-10, 1 - 1e-10] to avoid log(0)
    embeddings = np.clip(gmm_predictions, epsilon, 1.0 - epsilon)
    # Compute cross entropy
    num_samples = embeddings.shape[0]
    loss = -np.sum(np.log(gmm_predictions[np.arange(num_samples), gts])) / num_samples
    return torch.from_numpy(loss)


def get_optimizer(net: nn.Module):
    optimizer = optim.SGD(net.parameters(), lr=0.0003, momentum=0.9)
    return optimizer


class MyModel(nn.Module):
  def __init__(self, gmm: MyGaussianMixture, loss_function=None):
    super(MyModel, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
    self.lin1 = nn.Linear(9216, 128)
    self.lin2 = nn.Linear(128, 32)

    self.gmm = gmm
    self.loss_function = loss_function

  def forward(self, x, gts=None):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = torch.flatten(x, 1)
    x = self.lin1(x)
    x = F.relu(x)
    x = self.lin2(x)

    if self.training:
      self.gmm.fit(x)
      loss = self.loss_function(x, gts)
      return loss
    return x
  
  
def train_model(
    model: nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.SGD,
    loss_graph: List[float],
):
  model.train()

  for batch_id, (data, target) in enumerate(train_loader):
    data = data.to(device)
    target = target.to(device)

    optimizer.zero_grad()
    loss = model(data, target)
    loss_graph.append(loss.item())
    loss.backward()
    optimizer.step()

  return loss, loss_graph


def validate_model(
    model: nn.Module,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader
):
  val_loss = 0

  model.train(False)

  with torch.no_grad():
      for batch_id, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)

        softmax_outputs = model(data)
        val_loss += MyLossFunction(softmax_outputs, target)

  val_loss /= len(val_loader.dataset)
  return val_loss
