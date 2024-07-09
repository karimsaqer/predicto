import torch
import torch.nn as nn



class MIMBlock(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(MIMBlock, self).__init__()

        self.convlstm_c = None
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.height = height
        self.width = width
        self.layer_norm = layer_norm
        self.stride = stride
        self.in_channel = in_channel
        self.filter_size = filter_size
        self._forget_bias = 1.0

        self.build_MIMBlock_network()


    def build_MIMBlock_network(self):
        self.ct_weight = nn.Parameter(torch.zeros(self.num_hidden * 2, self.height, self.width))
        self.oc_weight = nn.Parameter(torch.zeros(self.num_hidden, self.height, self.width))

        # Initialize convolutional blocks
        self.conv_t_cc = self._init_conv_block(self.in_channel, self.num_hidden * 3, self.height, self.width, self.filter_size, self.stride, self.layer_norm)
        self.conv_s_cc = self._init_conv_block(self.num_hidden, self.num_hidden * 4, self.height, self.width, self.filter_size, self.stride, self.layer_norm)
        self.conv_x_cc = self._init_conv_block(self.num_hidden, self.num_hidden * 4, self.height, self.width, self.filter_size, self.stride, self.layer_norm)
        self.conv_h_concat = self._init_conv_block(self.num_hidden, self.num_hidden * 4, self.height, self.width, self.filter_size, self.stride, self.layer_norm)
        self.conv_x_concat = self._init_conv_block(self.num_hidden, self.num_hidden * 4, self.height, self.width, self.filter_size, self.stride, self.layer_norm)
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

    def MIMS(self, x, h_t, c_t):
        """
        MIMS module to compute the next hidden and cell state.
        """
        if h_t is None:
            h_t = self._init_state(x)
        if c_t is None:
            c_t = self._init_state(x)

        h_concat = self.conv_h_concat(h_t)
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        ct_activation = torch.mul(c_t.repeat(1, 2, 1, 1), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x is not None:
            x_concat = self.conv_x_concat(x)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)

            i_ = i_ + i_x
            f_ = f_ + f_x
            g_ = g_ + g_x
            o_ = o_ + o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = torch.mul(c_new, self.oc_weight)
        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new

    def forward(self, x, diff_h, h, c, m):
        """
        Forward pass for the MIMBlock.
        """
        # Initialize hidden, cell, memory, and diff_h states if they are None
        h = self._init_state(x) if h is None else h
        c = self._init_state(x) if c is None else c
        m = self._init_state(x) if m is None else m
        diff_h = self._init_state(x) if diff_h is None else diff_h

        # Compute convolutional layers for t, s, and x
        t_cc = self.conv_t_cc(h)
        s_cc = self.conv_s_cc(m)
        x_cc = self.conv_x_cc(x)

        # Split the outputs into respective gates
        i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, dim=1)
        i_t, g_t, o_t = torch.split(t_cc, self.num_hidden, dim=1)
        i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, dim=1)

        # Compute input, forget, output gates and cell update
        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
        o = torch.sigmoid(o_x + o_t + o_s)
        m_next = f_ * m + i_ * g_

        # Compute MIMS module
        c, self.convlstm_c = self.MIMS(diff_h, c, self.convlstm_c if self.convlstm_c is None else self.convlstm_c.detach())

        c_next = c + i * g
        cell = torch.cat((c_next, m_next), 1)
        h_next = o * torch.tanh(self.conv_last(cell))

        return h_next, c_next, m_next