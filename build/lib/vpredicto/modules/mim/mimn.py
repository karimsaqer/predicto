import torch
import torch.nn as nn


class MIMN(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(MIMN, self).__init__()

        # Initialize class variables
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.height = height
        self.width = width
        self.layer_norm = layer_norm
        self.stride = stride
        self.in_channel = in_channel
        self.filter_size = filter_size
        self.build_MIMN_network()

    def build_MIMN_network(self):
        # Initialize weights for cell and output gate modulation
        self.ct_weight = nn.Parameter(torch.zeros(self.num_hidden * 2, self.height, self.width))
        self.oc_weight = nn.Parameter(torch.zeros(self.num_hidden, self.height, self.width))

        # Build the convolutional layers
        self.conv_h_concat = self._init_conv_block(self.in_channel, self.num_hidden * 4, self.height, self.width, self.filter_size, self.stride, self.layer_norm)
        self.conv_x_concat = self._init_conv_block(self.in_channel, self.num_hidden * 4, self.height, self.width, self.filter_size, self.stride, self.layer_norm)
        self.conv_last = nn.Conv2d(self.num_hidden * 2, self.num_hidden, kernel_size=1, stride=1, padding=0, bias=False)
    def _init_conv_block(self, in_channels, out_channels, height, width, filter_size, stride, layer_norm):
        """
        Helper function to initialize a convolutional block with optional layer normalization.
        """
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)]
        if layer_norm:
            layers.append(nn.LayerNorm([out_channels, height, width]))
        return nn.Sequential(*layers)

    def _init_state(self, inputs):
        """
        Helper function to initialize the hidden or cell state to zeros.
        """
        return torch.zeros_like(inputs)

    def _compute_gates(self, x_concat, h_concat, c_t):
        """
        Compute the gates for the input and hidden states.
        """
        # Split the concatenated input and hidden states into their respective gates
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_c, f_c = torch.split(torch.mul(c_t.repeat(1, 2, 1, 1), self.ct_weight), self.num_hidden, dim=1)

        # Combine input and hidden gates with cell gates
        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        # If x is not None, add its contribution to the gates
        if x_concat is not None:
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
            i_ = i_ + i_x
            f_ = f_ + f_x
            g_ = g_ + g_x
            o_ = o_ + o_x

        return torch.sigmoid(i_), torch.sigmoid(f_ + self._forget_bias), torch.tanh(g_), o_

    def forward(self, x, h_t=None, c_t=None):
        """
        Forward pass for the MIMN cell.
        """
        # Initialize hidden and cell states if they are None
        if h_t is None:
            h_t = self._init_state(x)
        if c_t is None:
            c_t = self._init_state(x)

        # Compute concatenated hidden and input states
        h_concat = self.conv_h_concat(h_t)
        x_concat = self.conv_x_concat(x) if x is not None else None

        # Compute gates
        i_, f_, g_, o_ = self._compute_gates(x_concat, h_concat, c_t)

        # Update cell state
        c_new = f_ * c_t + i_ * g_

        # Compute output gate modulation
        o_c = torch.mul(c_new, self.oc_weight)

        # Update hidden state
        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new