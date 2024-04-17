from src.classifier import *
from src.model import *
from src.utils import *

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def __validate_classifier__(nn_model = None):

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

  # If you are NOT using Windows, set NUM_WORKERS to anything you want, e.g. NUM_WORKERS = 4,
  # but Windows has issues with multi-process dataloaders, so NUM_WORKERS must be 0 for Windows.
  NUM_WORKERS = 0

  val_loader = DataLoader(mnist_test, batch_size=1, num_workers=NUM_WORKERS, shuffle=False)


  if not nn_model:
    print("Loading classifier model from save...")
    classifier_model = ClassifierModel(device, None)
    classifier_model.load_state_dict(torch.load("./saves/classifier_model.pth"))
  else:
    classifier_model = nn_model

  classifier_model = classifier_model.to(device)

  val_loss, accurate, confusion_matrix = validate_classifier(classifier_model, device, val_loader)

  print(f"Validation loss: {val_loss}")
  print(f"Accuracy %: {accurate * 100}%")

  disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[i for i in range(10)])
  disp.plot()
  plt.title("Confusion Matrix")
  plt.show()


if __name__ == "__main__":
  __validate_classifier__()