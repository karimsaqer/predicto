import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminator Network: Classifies frames or sequences as real or fake.

    Inputs:
    - input_nc: Number of input channels (e.g., 1 for single frame discriminator, 10 for sequence discriminator).

    Structure:
    - Several convolutional layers with instance normalization and LeakyReLU activation.
    - Final layer outputs a single value representing real or fake classification.
    """
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),

            # One issue with the standard ReLU activation function is that it can cause "dead neurons."
            # This happens when the input to a ReLU neuron is always negative, causing it to output zero for all inputs.
            # Once a neuron dies, it can stop learning entirely because the gradient flowing through it will also be zero.
            # LeakyReLU mitigates this problem by allowing a small, non-zero gradient when the input is negative.

            nn.LeakyReLU(0.2, inplace=True)
        ]
        model += [
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        model += [
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        model += [
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        model += [nn.Conv2d(512, 1, kernel_size=4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)