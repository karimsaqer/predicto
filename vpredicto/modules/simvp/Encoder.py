import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, C_in: int, C_hid: int, N_S: int):
        super(Encoder, self).__init__()
        '''
        C_in == Number of input channels (e.g., 3 for RGB images)
        C_hid == Number of hidden channels (feature maps) in the encoder
        N_S == Number of convolutional layers in the encoder
        '''

        # Initialize an empty sequential container to hold the layers of the encoder
        self.enc_layers = nn.Sequential()

        # Define the pattern of strides: alternating between 1 and 2 for each layer
        '''
        Stride of 1: Keeps the spatial dimensions the same while applying the convolution operation. This is useful for extracting features without downsampling.
        Stride of 2: Reduces the spatial dimensions by half, effectively performing a downsampling operation. This is useful for capturing more global features and reducing the size of the feature maps.
        '''
        strides = [1 if i % 2 == 0 else 2 for i in range(N_S)]

        # Loop to create and add each layer to the encoder
        for i in range(N_S):
            if i == 0:
                # First layer: transforms the input channels to hidden channels
                self.enc_layers.add_module(f'conv_{i}', nn.Sequential(
                    nn.Conv2d(C_in, C_hid, kernel_size=3, stride=strides[i], padding=1),
                    nn.GroupNorm(2, C_hid),    # 2 ==> number of groups
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            else:
                # Subsequent layers: maintain the number of hidden channels
                self.enc_layers.add_module(f'conv_{i}', nn.Sequential(
                    nn.Conv2d(C_hid, C_hid, kernel_size=3, stride=strides[i], padding=1),
                    nn.GroupNorm(2, C_hid),
                    nn.LeakyReLU(0.2, inplace=True)
                ))

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass for the Encoder.

        Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
        tuple: A tuple containing the final output tensor and the output of the first layer
        """

        # Initialize the variable to store the output of the first layer
        first_layer_output = None

        # Process the input tensor through each layer of the encoder
        for i, layer in enumerate(self.enc_layers):
            x = layer(x)
            # Store the output of the first layer
            if i == 0:
                first_layer_output = x

        # Return the final output and the output of the first layer
        return x, first_layer_output