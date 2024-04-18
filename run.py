from argparse import ArgumentParser

from src.trainers import train_classifier
from src.trainers import train_encoder
from src.trainers import train_encoder_classifier
from src.validators import validate_classifier
from src.validators import validate_encoder
from src.validators import validate_encoder_classifier


TRAINERS = {
  "train_classifier": train_classifier.__train_classifier__,
  "train_encoder": train_encoder.__train_encoder__,
  "train_encoder_classifier": train_encoder_classifier.__train_encoder_classifier__,
}

VALIDATORS = {
  "validate_encoder": validate_encoder.__validate_encoder__,
  "validate_classifier": validate_classifier.__validate_classifier__,
  "validate_encoder_classifier": validate_encoder_classifier.__validate_encoder_classifier__,
}

MODEL_NAMES = ("classifier", "encoder", "encoder_classifier")


def main():
  parser = ArgumentParser(
    prog="run",
    description="Driver file for trainers and validators"
  )
  parser.add_argument("model", metavar="model", type=str, nargs=1, help=f"Model name. Valid options are: {', '.join(MODEL_NAMES)}")
  parser.add_argument("labeled_percent", metavar="labeled_percent", type=str, nargs=1, help="Percent of labeled training data. Write as a float (e.g. 0.8)")
  parser.add_argument("-t", "--train", action="store_true", help="Trains the model. Prompts the user to save the model after training.")
  parser.add_argument("-e", "--epochs", type=int, default=5, help="Number of epochs (default 1). If using a smaller labeled_percent, run with more epochs.")
  parser.add_argument("-v", "--val", action="store_true", help="Validates the model. If -t is set, uses the same model regardless if it was saved.")

  args = parser.parse_args()
  model = args.model[0]
  assert(model in MODEL_NAMES)

  labeled_percent = args.labeled_percent[0]
  labeled_percent_parsed = float(labeled_percent)
  assert(0 <= labeled_percent_parsed <= 1)
  percent_tag = labeled_percent.split(".")[1]

  epochs = args.epochs
  assert(epochs >= 1)
  train = args.train
  val = args.val

  nn_model = None

  if train:
    train_func = TRAINERS.get(f"train_{model}", None)
    if train_func is None:
      print(f"train_{model} does not exist!")
      return
    nn_model = train_func(epochs, labeled_percent_parsed, percent_tag)

  if val:
    val_func = VALIDATORS.get(f"validate_{model}", None)
    if val_func is None:
      print(f"validate_{model} does not exist!")
      return
    val_func(nn_model, percent_tag)
    input("Press any key to continue... (this is so the matplotlib figures don't close immediately)")


if __name__ == "__main__":
  main()
