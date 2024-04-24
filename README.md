# Installation

## **_Use Python 3.11.x_**

1. `python -m venv venv`
2. `./venv/Scripts/activate`
3. `mkdir ./saves`

If you are using CUDA with an nVidia GPU: `pip install -r requirements.txt`

If you are using CPU: `pip install -r requirements_cpu.txt`

# How to use run.py

To get help for command line arguments, run `python run.py --help`.
```
usage: run [-h] [-t] [-e EPOCHS] [-v] model labeled_percent

Driver file for trainers and validators

positional arguments:
  model                 Model name. Valid options are: classifier, encoder, encoder_classifier
  labeled_percent       Percent of labeled training data. Write as a float (e.g. 0.8)

options:
  -h, --help            show this help message and exit
  -t, --train           Trains the model. Prompts the user to save the model after training.
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs (default 1). If using a smaller labeled_percent, run with more epochs.
  -v, --val             Validates the model. If -t is set, uses the same model regardless if it was saved.
```

Example usages:
1. To train the encoder model with 80% training data and 15 epochs, `python run.py encoder 0.8 -t -e 15`
2. To train AND validate the encoder model with 80% training data and 15 epochs, `python run.py encoder 0.8 -tv -e 15`
3. To validate a saved encoder model with 80% training data and 15 epochs, `python run.py encoder 0.8 -v`
