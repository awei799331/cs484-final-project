import torch.nn as nn
import torch.optim as optim


def get_optimizer(net: nn.Module, lr: float, type="adam"):
  if type == "adam":
    optimizer = optim.Adam(net.parameters(), lr=lr)
  elif type == "sgd":
    optimizer = optim.SGD(net.parameters, lr=lr, momentum=0.9)
  return optimizer
