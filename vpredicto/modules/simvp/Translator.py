import torch
import torch.nn as nn
from .Inception import Inception

class Translator(nn.Module):
    def __init__(self, channel_in: int, channel_hid: int, N_T: int, incep_ker: list = [3, 5, 7, 11], groups: int = 8):
        super(Translator, self).__init__()
        '''
        channel_in == Number of input channels
        channel_hid == Number of hidden channels
        N_T == Number of Inception blocks in the translator
        incep_ker == List of kernel sizes for Inception blocks
        groups == Number of groups to separate the channels in GroupNorm
        '''

        self.N_T = N_T

        # Encoder Inception blocks
        self.enc_blocks = nn.ModuleList([
            Inception(
                C_in=channel_in if i == 0 else channel_hid,
                C_hid=channel_hid // 2,
                C_out=channel_hid,
                kernel_sizes=incep_ker,
                groups=groups
            )
            for i in range(N_T)
        ])

        # Decoder Inception blocks
        self.dec_blocks = nn.ModuleList([
            Inception(
                C_in=channel_hid if i == 0 else 2 * channel_hid,
                C_hid=channel_hid // 2,
                C_out=channel_in if i == N_T - 1 else channel_hid,
                kernel_sizes=incep_ker,
                groups=groups
            )
            for i in range(N_T)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass through the Translator module.
        x: Input tensor of shape (B, T, C, H, W)
        B: Batch size
        T: Temporal dimension (number of frames)
        C: Number of input channels
        H: Height of input
        W: Width of input
        '''

        # Reshape input to (B, T*C, H, W) to combine temporal and channel dimensions
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        skip_connections = []
        hidden = x

        # Forward pass through encoder blocks
        for i in range(self.N_T):
            hidden = self.enc_blocks[i](hidden)
            if i < self.N_T - 1:
                skip_connections.append(hidden)

        # Forward pass through decoder blocks with skip connections
        for i in range(self.N_T):
            if i > 0:
                hidden = torch.cat([hidden, skip_connections[-i]], dim=1)
            hidden = self.dec_blocks[i](hidden)

        # Reshape output back to (B, T, C, H, W)
        output = hidden.reshape(B, T, C, H, W)

        return output