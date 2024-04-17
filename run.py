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


def main():
  parser = ArgumentParser(
    prog="run",
    description="Driver file for trainers and validators"
  )
  parser.add_argument("model", metavar="S", type=str, nargs=1)
  parser.add_argument('-t', '--train', action='store_true')
  parser.add_argument('-v', '--val', action='store_true')

  args = parser.parse_args()
  model = args.model[0]
  train = args.train
  val = args.val

  nn_model = None

  if train:
    train_func = TRAINERS.get(f"train_{model}", None)
    if train_func is None:
      print(f"train_{model} does not exist!")
      return
    nn_model = train_func()

  if val:
    val_func = VALIDATORS.get(f"validate_{model}", None)
    if val_func is None:
      print(f"validate_{model} does not exist!")
      return
    val_func(nn_model)
    input("Press any key to continue... (this is so the matplotlib figures don't close immediately)")


if __name__ == "__main__":
  main()
