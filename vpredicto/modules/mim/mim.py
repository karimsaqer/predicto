import torch
import torch.nn as nn
import torch.nn.functional as F
from .stlstm import STLSTMCell
from .mimn import MIMN
from .mimblock import MIMBlock


class MIM_Model(nn.Module):
    def __init__(self , configs):
        super(MIM_Model, self).__init__()
        T, C, H, W = configs["in_shape"]

        self.configs = configs
        self.frame_channel = configs["patch_size"] ** 2 * C
        self.num_layers = len(configs["num_hidden"])
        self.num_hidden = configs["num_hidden"]
        self.height = H // configs["patch_size"]
        self.width = W // configs["patch_size"]
        self.MSE_criterion = nn.MSELoss()

        # Initialize ST-LSTM layers
        self.stlstm_layer = nn.ModuleList([
            self._init_stlstm_layer(i , configs) for i in range(self.num_layers)
        ])

        # Initialize differential ST-LSTM layers
        self.stlstm_layer_diff = nn.ModuleList([
            MIMN(self.num_hidden[i], self.num_hidden[i+1], self.height, self.width, configs["filter_size"],
                 configs["stride"], configs["layer_norm"]) for i in range(self.num_layers - 1)
        ])

        # Last convolution layer to generate the output frame
        self.conv_last = nn.Conv2d(self.num_hidden[self.num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        print(self)
    def _init_stlstm_layer(self, i, configs):
        """
        Initialize STLSTMCell or MIMBlock based on the layer index.
        """
        # print("here")
        in_channel = self.frame_channel if i == 0 else self.num_hidden[i - 1]
        if i < 1:
            return STLSTMCell(in_channel, self.num_hidden[i], self.height, self.width,
                              configs["filter_size"], configs["stride"], configs["layer_norm"])
        else:
            return MIMBlock(in_channel, self.num_hidden[i], self.height, self.width,
                            configs["filter_size"], configs["stride"], configs["layer_norm"])

    def _init_hidden_states(self, batch, device):
        """
        Initialize hidden and cell states to zeros.
        """
        zeros = [torch.zeros([batch, num_hidden, self.height, self.width], device=device)
                 for num_hidden in self.num_hidden]
        return zeros, zeros, [None] * self.num_layers, [None] * (self.num_layers - 1)

    def forward(self, frames_tensor, mask_true, return_loss=True):
        """
        Forward pass of the MIM model.
        """
        device = frames_tensor.device
        # Rearrange tensor dimensions
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        # print(frames.shape)
        batch = frames.shape[0]
        next_frames = []
        h_t, c_t, hidden_state_diff, cell_state_diff = self._init_hidden_states(batch, device)
        st_memory = torch.zeros([batch, self.num_hidden[0], self.height, self.width], device=device)

        for t in range(self.configs["in_frames_length"] + self.configs["out_frames_length"] - 1):
            # Select input frame or generated frame based on the time step
            net = frames[:, t] if t < self.configs["in_frames_length"] else \
                mask_true[:, t - self.configs["in_frames_length"]] * frames[:, t] + \
                (1 - mask_true[:, t - self.configs["in_frames_length"]]) * x_gen

            preh = h_t[0]
            # Forward pass through the first ST-LSTM layer
            h_t[0], c_t[0], st_memory = self.stlstm_layer[0](net, h_t[0], c_t[0], st_memory)

            for i in range(1, self.num_layers):
                # Differential ST-LSTM layers
                if t > 0:
                    if i == 1:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](
                            h_t[i - 1] - preh, hidden_state_diff[i - 1], cell_state_diff[i - 1])
                    else:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](
                            hidden_state_diff[i - 2], hidden_state_diff[i - 1], cell_state_diff[i - 1])
                else:
                    self.stlstm_layer_diff[i - 1](torch.zeros_like(h_t[i - 1]), None, None)

                # Forward pass through the subsequent ST-LSTM layers
                h_t[i], c_t[i], st_memory = self.stlstm_layer[i](
                    h_t[i - 1], hidden_state_diff[i - 1], h_t[i], c_t[i], st_memory)

            # Generate the output frame
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            # print(x_gen.shape)
            next_frames.append(x_gen)

        # Rearrange tensor dimensions back
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()

        # Calculate loss if required
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:]) if return_loss else None

        return next_frames, loss