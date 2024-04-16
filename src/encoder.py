from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class EncoderModel(nn.Module):
  def __init__(self, device: torch.device, loss_function=None):
    super(EncoderModel, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
    self.convt2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
    self.convt1 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)

    self.device = device
    self.loss_function = loss_function

  def forward(self, x: torch.Tensor, gts=None):
    y = self.encode(x)
    z = self.decode(y)

    if self.training:
      loss = self.loss_function(z, x)
      return loss
    return z
  
  def encode(self, x: torch.Tensor):
    y = self.conv1(x)
    y = F.relu(y)
    y = self.conv2(y)
    y = F.relu(y)
    y = self.conv3(y)
    y = F.relu(y)
    return y
  
  def decode(self, y: torch.Tensor):
    z = self.convt3(y)
    z = F.relu(z)
    z = self.convt2(z)
    z = F.relu(z)
    z = self.convt1(z)
    z = F.relu(z)
    return z


def train_encoder(
    model: nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.SGD,
    loss_graph: List[float],
    print_losses: bool = True
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

    if print_losses:
      print(loss_graph[-1])

  return loss


def validate_encoder(
    model: nn.Module,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
    save_first_n: int = 0,
):
  val_loss = 0

  input_data = []
  output_data = []

  model.train(False)

  with torch.no_grad():
      for batch_id, (data, target) in enumerate(val_loader):
        data = data.to(device)
        # target = target.to(device)

        encode_decode = model(data)
        val_loss += nn.MSELoss()(encode_decode, data).item()

        if batch_id < save_first_n:
          input_data.extend(data.cpu().detach())
          output_data.extend(encode_decode.cpu().detach())

  val_loss /= len(val_loader.dataset)
  return val_loss, input_data, output_data
