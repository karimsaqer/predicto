import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, C_hid: int, C_out: int, N_S: int):
        super(Decoder, self).__init__()
        '''
        C_hid == Number of hidden channels (feature maps) in the decoder
        C_out == Number of output channels (e.g., 3 for RGB images)
        N_S == Number of transposed convolutional layers in the decoder
        '''

        # Initialize an empty sequential container to hold the layers of the decoder
        self.dec_layers = nn.Sequential()

        # Define the pattern of strides: alternating between 1 and 2 for each layer (in reverse order)
        '''
        Stride of 1: Keeps the spatial dimensions the same while applying the transposed convolution operation.
        Stride of 2: Increases the spatial dimensions by a factor of 2, effectively performing an upsampling operation.
        '''
        strides = list(reversed([1 if i % 2 == 0 else 2 for i in range(N_S)]))

        # Loop to create and add each layer to the decoder
        for i in range(N_S-1):
            self.dec_layers.add_module(f'deconv_{i}', nn.Sequential(
                nn.ConvTranspose2d(C_hid, C_hid, kernel_size=3, stride=strides[i], padding=1, output_padding=(strides[i] - 1)),
                nn.GroupNorm(2, C_hid),
                nn.LeakyReLU(0.2, inplace=True)
            ))

        # Last layer: transforms the hidden channels to the hidden channels and includes a skip connection
        self.dec_layers.add_module(f'deconv_{N_S-1}', nn.Sequential(
            nn.ConvTranspose2d(2 * C_hid, C_hid, kernel_size=3, stride=strides[-1], padding=1, output_padding=(strides[-1] - 1)),
            nn.GroupNorm(2, C_hid),
            nn.LeakyReLU(0.2, inplace=True)
        ))

        # Final readout layer to get the desired output channels
        self.final_conv = nn.Conv2d(C_hid, C_out, kernel_size=1)

    def forward(self, hidden: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        '''
        hidden: Hidden tensor from the encoder or intermediate processing
        skip: Output of the first layer of the encoder (skip connection)
        '''
        # Pass the hidden tensor through all but the last layer
        for layer in self.dec_layers[:-1]:
            hidden = layer(hidden)

        # Concatenate the hidden tensor and the skip connection along the channel dimension
        combined = torch.cat([hidden, skip], dim=1)

        # Pass the combined tensor through the last layer
        hidden = self.dec_layers[-1](combined)

        # Apply the final convolution to get the output tensor
        output = self.final_conv(hidden)

        return output