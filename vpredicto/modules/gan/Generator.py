import torch
import torch.nn as nn
from .ResidualBlock import ResidualBlock

class Generator(nn.Module):
    """
    Generator Network: Generates future frames from input frames.

    Inputs:
    - input_nc: Number of input channels (e.g., 10 frames with 1 channel each, so 10).
    - output_nc: Number of output channels (e.g., 10 frames with 1 channel each, so 10).
    - n_residual_blocks: Number of residual blocks to use (default is 9).

    Structure:
    - Initial convolution block for feature extraction.
    - Downsampling layers to reduce spatial dimensions.
    - Residual blocks for deeper feature extraction.
    - Upsampling layers to restore spatial dimensions.
    - Output layer to generate the final frame output.
    """
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [
            nn.Conv2d(input_nc, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling:  Stride of 2: Reduces the spatial dimensions by half. Padding of 1: Ensures that the spatial dimensions are halved smoothly without losing too much information at the borders.
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling: upsampling is achieved using transposed convolutional layers (also known as deconvolutional layers). Each transposed convolution operation doubles the spatial dimensions (height and width) of the input feature maps while reducing the number of feature channels.
        # Stride of 2: Doubles the spatial dimensions.
        # Padding of 1: Ensures smooth upsampling without introducing artifacts at the borders.
        # Output padding of 1: Adjusts the output dimensions to match the desired spatial dimensions exactly.

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.Conv2d(64, output_nc, kernel_size=7, padding=3)]

        # Tanh function has a smooth gradient, which helps in stabilizing the training of the GAN. Smooth gradients facilitate better gradient flow during backpropagation, reducing the likelihood of encountering vanishing or exploding gradients.
        # Using tanh can lead to better convergence properties compared to other activation functions like sigmoid. The output range of [-1,1] is wider gradient range than [0,1] which helps in more stable and faster training.
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)