from src.encoder import *
from src.loader import *
from src.model import *
from src.utils import *

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


def __train_encoder__(epochs: int = 1, labeled_percent: float = 1.0, percent_tag: str = ""):

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  print(device)

  set_seed()

  dataset_handler = LabeledUnlabeledMNIST(labeled_percent)
  mnist_train = dataset_handler.labeled_dataset

  # Increase TRAIN_BATCH_SIZE if you are using GPU to speed up training. 
  # When batch size changes, the learning rate may also need to be adjusted. 
  # Note that batch size maybe limited by your GPU memory, so adjust if you get "run out of GPU memory" error.
  TRAIN_BATCH_SIZE = 100

  # If you are NOT using Windows, set NUM_WORKERS to anything you want, e.g. NUM_WORKERS = 4,
  # but Windows has issues with multi-process dataloaders, so NUM_WORKERS must be 0 for Windows.
  NUM_WORKERS = 0


  train_loader = DataLoader(mnist_train, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

  mse_loss = nn.MSELoss()
  mse_loss = mse_loss.to(device)

  encoder_model = EncoderModel(device, mse_loss)
  encoder_model = encoder_model.to(device)

  adam_optimizer = get_optimizer(encoder_model, 1e-4)

  loss_graph = []

  final_loss = train_encoder(
    encoder_model,
    device,
    train_loader,
    adam_optimizer,
    loss_graph,
    print_losses=False,
    epochs=epochs,
  )

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
    torch.save(encoder_model.state_dict(), f"./saves/encoder_model_{percent_tag}.pth")
  else:
    print("Model not saved. Exiting.")


if __name__ == "__train_encoder__":
  train_encoder()