from src.classifier import *
from src.model import *
from src.utils import *

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def __train_classifier__():

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  print(device)

  """
  Checks the filepath "./data/MNIST/raw" for the dataset. If not found, downloads the dataset
  """
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
  ])

  mnist_train, mnist_test = download(transform)

  # Increase TRAIN_BATCH_SIZE if you are using GPU to speed up training. 
  # When batch size changes, the learning rate may also need to be adjusted. 
  # Note that batch size maybe limited by your GPU memory, so adjust if you get "run out of GPU memory" error.
  TRAIN_BATCH_SIZE = 100

  # If you are NOT using Windows, set NUM_WORKERS to anything you want, e.g. NUM_WORKERS = 4,
  # but Windows has issues with multi-process dataloaders, so NUM_WORKERS must be 0 for Windows.
  NUM_WORKERS = 0


  train_loader = DataLoader(mnist_train, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

  ce_loss = nn.CrossEntropyLoss()
  ce_loss = ce_loss.to(device)

  classifier_model = ClassifierModel(device, ce_loss)
  classifier_model = classifier_model.to(device)

  sgd_optimizer = get_optimizer(classifier_model, 1e-4)

  loss_graph = []

  final_loss = train_classifier(classifier_model, device, train_loader, sgd_optimizer, loss_graph)

  print(f"Completed training! Final loss: {final_loss.item()}\nRunning validation...")

  plt.clf()
  plt.cla()
  plt.plot(loss_graph)
  plt.ylabel("Cross Entropy Loss")
  plt.xlabel("Batch Number")
  plt.title("Training Loss Over Batch Number")
  plt.show()

  save_model = input("Save model? [y/n]... ")
  if save_model.lower() == "y":
    print("Saving model...")
    torch.save(classifier_model.state_dict(), './saves/classifier_model.pth')
  else:
    print("Model not saved. Exiting.")

  return classifier_model


if __name__ == "__main__":
  __train_classifier__()