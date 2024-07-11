import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Residual Block: Helps the generator learn the identity function.

    Inputs:
    - in_channels: Number of input channels for the block.

    Structure:
    - Two convolutional layers with instance normalization and ReLU activation.
    - Adds the input to the output to learn residuals.
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm2d(in_channels)  # Instance normalization normalizes the activations of each example in a batch independently, as opposed to batch normalization which normalizes across the entire batch. This is particularly useful in tasks where the appearance of the input (e.g., style in images) varies greatly between examples.
        self.relu = nn.ReLU(inplace=True)          # Adds non-linearities to the network, enabling the model to learn complex functions.
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out