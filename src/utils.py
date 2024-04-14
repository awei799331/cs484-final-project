import os
import shutil
from typing import Tuple

import torchvision.datasets as datasets

FILES = (
  "t10k-images-idx3-ubyte.gz",
  "t10k-labels-idx1-ubyte.gz",
  "train-images-idx3-ubyte.gz",
  "train-labels-idx1-ubyte.gz",
)

DATA_PATH = "./data/MNIST/raw"

def download() -> Tuple[datasets.VisionDataset, datasets.VisionDataset]:
  need_to_download = False

  if not os.path.exists(DATA_PATH):
    need_to_download = True

  for file in FILES:
    if not os.path.isfile(os.path.join(DATA_PATH, file)):
      need_to_download = True
      break

  if need_to_download:
    shutil.rmtree(DATA_PATH)

  mnist_train = download_mnist(train=True, download=need_to_download)
  mnist_test = download_mnist(train=False, download=need_to_download)

  return mnist_train, mnist_test


def download_mnist(train: bool, download: bool) -> datasets.VisionDataset:
  mnist_set = datasets.MNIST(root='./data', train=train, download=download, transform=None)
  return mnist_set
