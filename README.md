# vPredicto

**Predicto** is a Python library for video frame prediction, featuring three state-of-the-art models: PredRNN++, MIM, and Causal LSTM. This library is designed to cater to both expert and non-expert users, providing an API for developers and a simple interface for non-experts.

## Features

- Three video frame prediction models: PredRNN++, MIM, and Causal LSTM.
- Easy-to-use interface for training and testing models.
- Supports custom dataloaders or default to MovingMNIST dataset.
- Pre and post-processing for input and output in each model.

## Installation

```sh
pip install vpredicto
```

## Usage

## Quick Start
```sh
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

## Models
- PredRNN++: A recurrent neural network model for video frame prediction.
- MIM: Memory In Memory network for spatiotemporal predictive learning.
- Causal LSTM: A causal LSTM model for video frame prediction.