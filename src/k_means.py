from typing import List
import numpy as np
import torch
from sklearn.cluster import KMeans

class KMeansModel():
  def __init__(self, tensor_dims):
    self.tensor_dims = tensor_dims
    self.model = KMeans(n_clusters=10)
    self.xs: List[np.ndarray] = [None for _ in range(10)]
    self.initial_labelled_centroids = [None for _ in range(10)]
    self.all_points = None

  def add_xs(self, xs, ys):
    for i in range(10):
      self.xs[i] = xs[ys == i]
      self.initial_labelled_centroids[i] = np.mean(self.xs[i], axis=0)

  def clear(self):
    self.xs = None
    self.init = None

  def fit(self, xs, ys, unlabeled_encodings=None):
    if torch.is_tensor(xs):
      xs = xs.cpu().detach().numpy().reshape(-1, self.tensor_dims)
    
    if torch.is_tensor(ys):
      ys = ys.cpu().detach().numpy()

    self.add_xs(xs, ys)
    self.all_points = xs

    if unlabeled_encodings is not None:
      if torch.is_tensor(unlabeled_encodings):
        unlabeled_encodings = unlabeled_encodings.cpu().detach().numpy().reshape(-1, self.tensor_dims)
      self.all_points = np.concatenate((xs, unlabeled_encodings), axis=0)

    self.model = KMeans(n_clusters=10, init=self.initial_labelled_centroids)
    self.model.fit(self.all_points)

  def predict(self, xs):
    if torch.is_tensor(xs):
      xs = xs.cpu().detach().numpy().reshape(-1, self.tensor_dims)
    predictions = self.model.predict(xs)
    return predictions
  
  def predict_all_points(self):
    predictions = self.model.predict(self.all_points)
    return self.all_points, predictions
