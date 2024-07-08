<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predicto</title>
</head>
<body>
    <h1>Predicto</h1>
    <p><strong>Predicto</strong> is a Python library for video frame prediction, featuring three state-of-the-art models: PredRNN++, MIM, and Causal LSTM. This library is designed to cater to both expert and non-expert users, providing an API for developers and a simple interface for non-experts.</p>

    <h2>Features</h2>
    <ul>
        <li>Three video frame prediction models: PredRNN++, MIM, and Causal LSTM.</li>
        <li>Easy-to-use interface for training and testing models.</li>
        <li>Supports custom dataloaders or default to MovingMNIST dataset.</li>
        <li>Pre and post-processing for input and output in each model.</li>
    </ul>

    <h2>Installation</h2>
    <pre><code>pip install predicto</code></pre>

    <h2>Usage</h2>
    <h3>Quick Start</h3>
    <pre><code>from predicto import PredRNN, MIM, CausalLSTM, Predicto

# Create a model object
model_object = MIM()

# Initialize Predicto with the model object
model = Predicto(model_object)

# Train the model
model.train(train_loader)

# Test the model
model.test(test_loader)</code></pre>

    <h3>Custom Dataloader</h3>
    <pre><code>from predicto import PredRNN, MIM, CausalLSTM, Predicto

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
model.train(train_loader)

# Test the model
model.test(test_loader)</code></pre>

    <h2>Models</h2>
    <ul>
        <li><strong>PredRNN++</strong>: A recurrent neural network model for video frame prediction.</li>
        <li><strong>MIM</strong>: Memory In Memory network for spatiotemporal predictive learning.</li>
        <li><strong>Causal LSTM</strong>: A causal LSTM model for video frame prediction.</li>
    </ul>

</body>
</html>
