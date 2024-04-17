from typing import List

import numpy as np
from sklearn import mixture
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


class MyGaussianMixture(object):
  def __init__(self):
    self.mixture = mixture.GaussianMixture(n_components=10, warm_start=True)
    self.datapoints = None

  def clear(self):
    self.datapoints = None

  def fit(self, data: torch.Tensor):
    points = data.cpu().detach().numpy()
    if self.datapoints is None:
      self.datapoints = points
    else:
      self.datapoints = np.append(self.datapoints, points, axis=0)
    
    self.mixture.fit(self.datapoints)

  def predict_one(self, point: torch.Tensor):
    return self.mixture.predict_proba(point.cpu().detach().numpy()[np.newaxis, :])

  def predict_many(self, points: torch.Tensor):
    return self.mixture.predict_proba(points.cpu().detach().numpy())


def get_optimizer(net: nn.Module, lr: float):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    return optimizer
