from functools import cache
import os
import shutil
from typing import Tuple

import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
import torchvision.datasets as datasets

FILES = (
  "t10k-images-idx3-ubyte.gz",
  "t10k-labels-idx1-ubyte.gz",
  "train-images-idx3-ubyte.gz",
  "train-labels-idx1-ubyte.gz",
)

DATA_PATH = "./data/MNIST/raw"

def download(transform: Compose = None) -> Tuple[datasets.MNIST, datasets.MNIST]:
  need_to_download = False

  if not os.path.exists(DATA_PATH):
    need_to_download = True

  for file in FILES:
    if not os.path.isfile(os.path.join(DATA_PATH, file)):
      need_to_download = True
      break

  if os.path.exists(DATA_PATH) and need_to_download:
    shutil.rmtree(DATA_PATH)

  mnist_train = download_mnist(train=True, download=need_to_download, transform=transform)
  mnist_test = download_mnist(train=False, download=need_to_download, transform=transform)

  return mnist_train, mnist_test


def download_mnist(train: bool, download: bool, transform: Compose = None) -> datasets.MNIST:
  mnist_set = datasets.MNIST(
    root="./data",
    train=train,
    download=download,
    transform=transform,
  )
  return mnist_set

def set_seed(seed=484):
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

@cache
def get_transform():
  return Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
  ])