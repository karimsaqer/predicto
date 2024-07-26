# Predicto

**Predicto** is a Python library for video frame prediction, featuring three state-of-the-art models: PredRNN++, MIM, and Causal LSTM. This library is designed to cater to both expert and non-expert users, providing an API for developers and a simple interface for non-experts.

## Features
- Three video frame prediction models: PredRNN++, MIM, and Causal LSTM.
- Easy-to-use interface for training and testing models.
- Supports custom dataloaders or defaults to the MovingMNIST dataset.
- Pre- and post-processing for input and output in each model.

## Installation
```sh
pip install predicto
```

# Usage
## Quick Start
```python
from predicto import PredRNN, MIM, CausalLSTM, Predicto

# Create a model object
model_object = MIM()

# Initialize Predicto with the model object
model = Predicto(model_object)

# Train the model
model.train(train_loader)

# Test the model
model.test(test_loader)
```


## Custom Dataloader
```python
from predicto import PredRNN, MIM, CausalLSTM, Predicto

# Define your custom dataloader
class CustomDataLoader:
    def __init__(self, ...):
        ...

    def __iter__(self):
        ...

# Create a model object
model_object = CausalLSTM()

# Initialize Predicto with the model object and custom dataloader
model = Predicto(model_object, dataloader=CustomDataLoader())

# Train the model
model.train()

# Test the model
model.test()
```

## Models
* SimVP
 - PredRNN++
  * MIM
 * PredNet
 * Novel GAN
 * Causal LSTM.