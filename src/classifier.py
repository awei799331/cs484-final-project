from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class ClassifierModel(nn.Module):
  def __init__(self, device: torch.device, loss_function=None):
    super(ClassifierModel, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
    self.lin1 = nn.Linear(9216, 128)
    self.lin2 = nn.Linear(128, 10)

    self.device = device
    self.loss_function = loss_function

  def forward(self, x: torch.Tensor, gts=None):
    y = self.conv1(x)
    y = F.relu(y)
    y = self.conv2(y)
    y = F.relu(y)
    y = F.max_pool2d(y, 2)
    y = torch.flatten(y, 1)
    y = self.lin1(y)
    y = F.relu(y)
    y = self.lin2(y)

    if self.training:
      loss = self.loss_function(y, gts)
      return loss
    return y


class EncoderClassifierModel(nn.Module):
  def __init__(self, device: torch.device, loss_function=None):
    super(EncoderClassifierModel, self).__init__()
    self.lin1 = nn.Linear(256, 128)
    self.lin2 = nn.Linear(128, 64)
    self.lin3 = nn.Linear(64, 32)
    self.lin4 = nn.Linear(32, 10)

    self.device = device
    self.loss_function = loss_function

  def forward(self, x: torch.Tensor, gts=None):
    y = torch.flatten(x, 1)
    y = self.lin1(y)
    y = F.relu(y)
    y = self.lin2(y)
    y = F.relu(y)
    y = self.lin3(y)
    y = F.relu(y)
    y = self.lin4(y)

    if self.training:
      loss = self.loss_function(y, gts)
      return loss
    return y


def train_classifier(
    model: nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_graph: List[float],
    print_losses: bool = False,
    epochs: int = 1,
):
  model.train()

  for i in range(epochs):
    for batch_id, (data, target) in enumerate(train_loader):
      data = data.to(device)
      target = target.to(device)

      optimizer.zero_grad()
      loss = model(data, target)
      loss_graph.append(loss.item())
      loss.backward()
      optimizer.step()

      if print_losses:
        print(loss_graph[-1])

    print(f"Loss at epoch {i}: {loss_graph[-1]}")

  return loss

def validate_classifier(
    model: nn.Module,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader
):
  val_loss = 0
  accurate = 0

  confusion_matrix = np.zeros((10, 10))

  model.train(False)

  with torch.no_grad():
      for batch_id, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        softmax_outputs = F.softmax(output, 1)
        val_loss += nn.CrossEntropyLoss()(softmax_outputs, target).item()
        prediction = softmax_outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        accurate += prediction.eq(target.view_as(prediction)).sum().item()

        for label, pred in zip(target, prediction):
          confusion_matrix[label, pred] += 1

  val_loss /= len(val_loader.dataset)
  accurate /= len(val_loader.dataset)
  return val_loss, accurate, confusion_matrix

def train_encoder_classifier(
    model: nn.Module,
    encoder_model: nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_graph: List[float],
    print_losses: bool = False,
    epochs: int = 1
):
  model.train()
  encoder_model.train(False)

  for i in range(epochs):
    for batch_id, (data, target) in enumerate(train_loader):
      data = data.to(device)
      target = target.to(device)

      data_encoded = encoder_model.encode(data)

      optimizer.zero_grad()
      loss = model(data_encoded, target)
      loss_graph.append(loss.item())
      loss.backward()
      optimizer.step()

      if print_losses:
        print(loss_graph[-1])

    print(f"Loss at epoch {i}: {loss_graph[-1]}")

  return loss

def validate_encoder_classifier(
    model: nn.Module,
    encoder_model: nn.Module,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader
):
  val_loss = 0
  accurate = 0

  confusion_matrix = np.zeros((10, 10))

  model.train(False)
  encoder_model.train(False)

  with torch.no_grad():
      for batch_id, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)

        data_encoded = encoder_model.encode(data)

        output = model(data_encoded)
        softmax_outputs = F.softmax(output, 1)
        val_loss += nn.CrossEntropyLoss()(softmax_outputs, target).item()
        prediction = softmax_outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        accurate += prediction.eq(target.view_as(prediction)).sum().item()

        for label, pred in zip(target, prediction):
          confusion_matrix[label, pred] += 1

  val_loss /= len(val_loader.dataset)
  accurate /= len(val_loader.dataset)
  return val_loss, accurate, confusion_matrix