import torch
import torch.nn as nn



class STLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(STLSTMCell, self).__init__()

        # Initialize class variables
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.in_channel = in_channel
        self.height = height
        self.width = width
        self.layer_norm = layer_norm
        self.stride = stride

        # Build the Spatio-Temporal LSTM network
        self.build_stlstm_network()

    def build_stlstm_network(self):
        """
        Initialize all convolutional layers for the STLSTM cell.
        """
        self.conv_x = self._init_conv_block(self.in_channel, self.num_hidden * 7, self.layer_norm)
        self.conv_h = self._init_conv_block(self.num_hidden, self.num_hidden * 4, self.layer_norm)
        self.conv_m = self._init_conv_block(self.num_hidden, self.num_hidden * 3, self.layer_norm)
        self.conv_o = self._init_conv_block(self.num_hidden * 2, self.num_hidden, self.layer_norm)
        self.conv_last = nn.Conv2d(self.num_hidden * 2, self.num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def _init_conv_block(self, in_channels, out_channels, layer_norm):
        """
        Helper function to initialize a convolutional block.
        """
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=self.padding * 2 + 1, stride=self.stride, padding=self.padding, bias=False)
        ]
        if layer_norm:
            layers.append(nn.LayerNorm([out_channels, self.height, self.width]))
        return nn.Sequential(*layers)

    def apply_conv(self, x, conv_layer):
        """
        Apply a convolutional layer to the input tensor.
        """
        return conv_layer(x)

    def compute_hidden_gates(self, h_t):
        """
        Compute gates for the hidden state.
        """
        h_concat = self.apply_conv(h_t, self.conv_h)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        return i_h, f_h, g_h, o_h

    def compute_input_x_gates(self, x_t):
        """
        Compute gates for the input tensor.
        """
        x_concat = self.apply_conv(x_t, self.conv_x)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        return i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x

    def compute_memory_gates(self, m_t):
        """
        Compute gates for the memory state.
        """
        m_concat = self.apply_conv(m_t, self.conv_m)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)
        return i_m, f_m, g_m

    def forward(self, x_t, h_t, c_t, m_t):
        """
        Forward pass for the STLSTM cell.
        """
        # Compute gates for input, hidden, and memory states
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = self.compute_input_x_gates(x_t)
        i_h, f_h, g_h, o_h = self.compute_hidden_gates(h_t)
        i_m, f_m, g_m = self.compute_memory_gates(m_t)

        # Compute the cell state
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
        c_next = f_t * c_t + i_t * g_t

        # Compute the memory state
        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)
        m_next = f_t_prime * m_t + i_t_prime * g_t_prime

        # Combine cell and memory states, compute the output gate and hidden state
        mem = torch.cat((c_next, m_next), 1)
        o_t = torch.sigmoid(o_x + o_h + self.apply_conv(mem, self.conv_o))
        h_next = o_t * torch.tanh(self.apply_conv(mem, self.conv_last))

        return h_next, c_next, m_next