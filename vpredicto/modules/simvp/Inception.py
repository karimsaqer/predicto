import torch
import torch.nn as nn

class Inception(nn.Module):
    def __init__(self, C_in: int, C_hid: int, C_out: int, kernel_sizes: list = [3, 5, 7, 11], groups: int = 8):
        super(Inception, self).__init__()
        '''
        C_in == Number of input channels
        C_hid == Number of hidden channels (intermediate channels before output)
        C_out == Number of output channels
        kernel_sizes == List of kernel sizes for each convolution operation
        groups == Number of groups to separate the channels in GroupNorm
        '''

        # 1x1 convolution to reduce the number of input channels to hidden channels
        self.conv1x1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)

        # List of convolutional layers with different kernel sizes
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C_hid, C_out, kernel_size=ker_size, stride=1, padding=ker_size // 2),
                nn.GroupNorm(groups, C_out),
                nn.LeakyReLU(0.2, inplace=True)
            ) for ker_size in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply 1x1 convolution to reduce input channels to hidden channels
        x = self.conv1x1(x)

        # Apply each convolutional layer with different kernel sizes and sum the outputs
        output = torch.stack([layer(x) for layer in self.conv_layers], dim=0).sum(dim=0)

        return output