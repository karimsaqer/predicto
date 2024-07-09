import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import os



class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_layer_dim, kernel_size, stride=1):
        super(ConvLSTMCell, self).__init__()

        # Initialize parameters
        self.input_dim = input_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.stride = stride
        self.build_conv_lstm_cell_network()

    def build_conv_lstm_cell_network(self):
      # Convolution layer for the input tensor
        self.input_conv = nn.Conv2d(
            self.input_dim, self.hidden_layer_dim * 4,
            kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, bias=False
        )

        # Convolution layer for the hidden state
        self.hidden_conv = nn.Conv2d(
            self.hidden_layer_dim, self.hidden_layer_dim * 4,
            kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, bias=False
        )

        # Convolution layer for combining input and hidden states
        self.output_conv = nn.Conv2d(
            self.hidden_layer_dim * 2, self.hidden_layer_dim,
            kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, bias=False
        )

        # Final convolution layer to generate the next hidden state
        self.final_conv = nn.Conv2d(
            self.hidden_layer_dim * 2, self.hidden_layer_dim,
            kernel_size=1, stride=1, padding=0, bias=False
        )


    # Method to apply convolution to the input tensor and the current hidden state
    def apply_convolution(self, input_frames_tensor, current_hidden):
        return self.input_conv(input_frames_tensor), self.hidden_conv(current_hidden)

    # Method to compute gates based on the convolutions of input and hidden state
    def compute_gates(self, input_frames_tensor, current_hidden):
        x_conv, h_conv = self.apply_convolution(input_frames_tensor, current_hidden)

        # Split the convolutions into four parts for the gates (input, forget, output, and cell gates)
        conv_i_x, conv_f_x, conv_o_x, conv_g_x = torch.split(x_conv, self.hidden_layer_dim, dim=1)
        conv_i_h, conv_f_h, conv_o_h, conv_g_h = torch.split(h_conv, self.hidden_layer_dim, dim=1)

        # Compute the input gate (i) by applying sigmoid to the sum of input and hidden convolutions
        i = torch.sigmoid(conv_i_x + conv_i_h)
        # Compute the forget gate (f) by applying sigmoid to the sum of input and hidden convolutions
        f = torch.sigmoid(conv_f_x + conv_f_h)
        # Compute the output gate (o) by applying sigmoid to the sum of input and hidden convolutions
        o = torch.sigmoid(conv_o_x + conv_o_h)
        # Compute the cell gate (g) by applying tanh to the sum of input and hidden convolutions
        g = torch.tanh(conv_g_x + conv_g_h)

        return i, f, o, g

    def forward(self, input_frames_tensor, cur_state):
        # Get the current hidden state and cell state from cur_state
        h_cur, c_cur = cur_state

        # Compute the gates
        input_gate, forget_gate, out_gate, g_gate = self.compute_gates(input_frames_tensor, h_cur)

        # Compute the next cell state (c_next) using the forget gate, current cell state, input gate, and cell gate
        c_next = forget_gate * c_cur + input_gate * g_gate

        # Compute the next hidden state (h_next) using the output gate and next cell state
        h_next = out_gate * torch.tanh(c_next)

        # Return the next hidden state and next cell state
        return h_next, c_next