import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt


'''
Causal LSTM Cell
__init__ method to initialize the Causal LSTM Cell: you can pass the input_channels, hidden_channels, and kernel_size as parameters
forward method to pass the input and hidden state through the Causal LSTM Cell
the input is the input and hidden state
the output is the updated hidden state, cell state, and memory state
'''
class CausalLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(CausalLSTMCell, self).__init__()

        self.hidden_channels = hidden_channels
        self.padding = kernel_size // 2
        self._forget_bias = 1.0

        self.conv_x = nn.Conv2d(input_channels, hidden_channels * 7, kernel_size=kernel_size, padding=self.padding, bias=False)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels * 4, kernel_size=kernel_size, padding=self.padding, bias=False)
        self.conv_c = nn.Conv2d(hidden_channels, hidden_channels * 3, kernel_size=kernel_size, padding=self.padding, bias=False)
        self.conv_m = nn.Conv2d(hidden_channels, hidden_channels * 3, kernel_size=kernel_size, padding=self.padding, bias=False)
        self.conv_o = nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=kernel_size, padding=self.padding, bias=False)
        self.conv_c2m = nn.Conv2d(hidden_channels, hidden_channels * 4, kernel_size=kernel_size, padding=self.padding, bias=False)
        self.conv_om = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=self.padding, bias=False)
        self.conv_last = nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_gate_CL = self.conv_x(x_t)
        h_gate_CL = self.conv_h(h_t)
        c_gate_CL = self.conv_c(c_t)
        m_gate_CL = self.conv_m(m_t)

        i_x, f_x, g_x, i_x_dash, f_x_dash, g_x_dash, o_x = torch.split(x_gate_CL, self.hidden_channels, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_gate_CL, self.hidden_channels, dim=1)
        i_m, f_m, m_m = torch.split(c_gate_CL, self.hidden_channels, dim=1)
        i_c, f_c, g_c = torch.split(c_gate_CL, self.hidden_channels, dim=1)

        i_t = torch.sigmoid(i_x + i_h + i_c)
        f_t = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
        g_t = torch.tanh(g_x + g_h + g_c)

        c_new = f_t * c_t + i_t * g_t

        c_to_m = self.conv_c2m(c_new)
        i_c, g_c, f_c, o_c = torch.split(c_to_m, self.hidden_channels, dim=1)

        i_t_dash = torch.sigmoid(i_x_dash + i_m + i_c)
        f_t_dash = torch.sigmoid(f_x_dash + f_m + f_c + self._forget_bias)
        g_t_dash = torch.tanh(g_x_dash + g_c)

        m_new = f_t_dash * torch.tanh(m_m) + i_t_dash * g_t_dash
        o_m = self.conv_om(m_new)

        o_t = torch.tanh(o_x + o_h + o_c + o_m)
        memory = torch.cat((c_new, m_new), 1)
        h_new = o_t * torch.tanh(self.conv_last(memory))

        return h_new, c_new, m_new