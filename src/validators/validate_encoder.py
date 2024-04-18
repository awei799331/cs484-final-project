
from src.encoder import *
from src.loader import *
from src.model import *
from src.utils import *

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


def __validate_encoder__(nn_model=None, percent_tag: str = ""):

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  print(device)

  set_seed()

  dataset_handler = LabeledUnlabeledMNIST()
  mnist_test = dataset_handler.mnist_test

  # If you are NOT using Windows, set NUM_WORKERS to anything you want, e.g. NUM_WORKERS = 4,
  # but Windows has issues with multi-process dataloaders, so NUM_WORKERS must be 0 for Windows.
  NUM_WORKERS = 0

  val_loader = DataLoader(mnist_test, batch_size=1, num_workers=NUM_WORKERS, shuffle=True)

  if not nn_model:
    print("Loading encoder model from save...")
    encoder_model = EncoderModel(device, None)
    encoder_model.load_state_dict(torch.load(f"./saves/encoder_model_{percent_tag}.pth"))
  else:
    encoder_model = nn_model
  
  encoder_model = encoder_model.to(device)

  encoder_model.train(False)

  save_first_n = 10
  val_loss, input_images, output_images = validate_encoder(encoder_model, device, val_loader, save_first_n)

  print(f"Validation loss: {val_loss}")

  plt.clf()
  plt.cla()
  fig, axs = plt.subplots(3, save_first_n)
  fig.set_figheight(10)
  fig.set_figwidth(50)

  for i in range(10):
    diff = (input_images[i].permute(1, 2, 0) - output_images[i].permute(1, 2, 0)).square()
    axs[0, i].imshow(input_images[i].permute(1, 2, 0), cmap="Grays")
    axs[0, i].set_title("Original")
    axs[0, i].axis("off")
    axs[1, i].imshow(output_images[i].permute(1, 2, 0), cmap="Grays")
    axs[1, i].set_title("Decoded")
    axs[1, i].axis("off")
    axs[2, i].imshow(diff, cmap="Grays", vmin=0, vmax=1)
    axs[2, i].set_title("Squared Diff")
    axs[2, i].axis("off")
  fig.show()


if __name__ == "__main__":
  __validate_encoder__()
