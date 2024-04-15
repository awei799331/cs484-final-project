from functools import cache

import numpy as np
from sklearn import mixture
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MyGaussianMixture():
  def __init__(self):
    self.mixture = mixture.GaussianMixture(n_components=10)

  def fit(self, data):
    self.mixture.fit(data)

  def predict_one(self, point):
    return self.mixture.predict_proba(point[np.newaxis, :])

  def predict_many(self, points):
    return self.mixture.predict_proba(points)


class MyLossFunction(nn.Module):
  def __init__(self, gmm: MyGaussianMixture):
    super(MyModel, self).__init__()
    self.gmm = gmm

  def forward(self, embeddings, gts):
    gmm_predictions = self.gmm.predict_many(embeddings)
    # Ensure numerical stability
    epsilon = 1e-10
    # Clip embeddings to avoid log(0)
    embeddings = np.clip(gmm_predictions, epsilon, 1.0 - epsilon)
    # Compute cross entropy
    num_samples = embeddings.shape[0]
    loss = -np.sum(np.log(gmm_predictions[np.arange(num_samples), gts])) / num_samples
    return loss


@cache
def get_optimizer(net):
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
