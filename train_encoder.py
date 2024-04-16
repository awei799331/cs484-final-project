from src.encoder import *
from src.model import *
from src.utils import *

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def __main__():

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
  val_loader = DataLoader(mnist_test, batch_size=1, num_workers=NUM_WORKERS, shuffle=False)

  my_gmm = MyGaussianMixture()
  my_gmm.clear()

  mse_loss = nn.MSELoss()
  mse_loss = mse_loss.to(device)

  encoder_model = EncoderModel(device, mse_loss)
  encoder_model = encoder_model.to(device)

  sgd_optimizer = get_optimizer(encoder_model, 1e-4)

  loss_graph = []

  final_loss = train_encoder(encoder_model, device, train_loader, sgd_optimizer, loss_graph)

  print(f"Completed training! Final loss: {final_loss.item()}\nRunning validation...")

  plt.clf()
  plt.cla()
  plt.plot(loss_graph)
  plt.ylabel("Cross Entropy Loss")
  plt.xlabel("Batch Number")
  plt.title("Training Loss Over Batch Number")
  plt.show()

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

  save_model = input("Save model? [y/n]... ")
  if save_model.lower() == "y":
    print("Saving model...")
    torch.save(encoder_model.state_dict(), './saves/encoder_model.pth')
  else:
    print("Model not saved. Exiting.")


if __name__ == "__main__":
  __main__()