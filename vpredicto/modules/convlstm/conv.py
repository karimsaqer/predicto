import torch
import torch.utils.data
import torch.nn as nn
from .cell import ConvLSTMCell


class ConvLSTM(nn.Module):
    def __init__(self , hidden_layer_dim = [128, 128, 128, 128] , configs = None):
        super(ConvLSTM, self).__init__()
        self.configs = configs or {
            'in_shape': [10, 1, 64, 64],
            'filter_size': 5,
            'stride': 1,
            'patch_size':2,
            'in_frames_length': 10,
            'out_frames_length': 10,
            "total_length":20,
            "batch_size" : 16,
            "r_sampling_step_1" : 25000,
            "r_sampling_step_2" : 50000,
            "r_exp_alpha" : 5000,
            "sampling_stop_iter" : 50000,
            "sampling_start_value" : 1.0,
            "sampling_changing_rate" : 0.00002,
            'device':"cuda"
        }
        frames_gif, channels, height, width = self.configs['in_shape']
        self.frame_channel = self.configs['patch_size'] ** 2 * channels
        self.num_layers = len(hidden_layer_dim)
        self.hidden_layer_dim = hidden_layer_dim
        self.in_frames_length = self.configs['in_frames_length']
        self.out_frames_length = self.configs['out_frames_length']
        self.total_frames_length = self.in_frames_length + self.out_frames_length
        self.MSE_criterion = nn.MSELoss()
        self.build_conv_lstm_network(self.configs['filter_size'])

        print(self)
    def build_conv_lstm_network(self,kernel_size):
        cell_list = []
        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else self.hidden_layer_dim[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channel, self.hidden_layer_dim[i], kernel_size)
            )
        self.cell_list = nn.ModuleList(cell_list)

        self.conv_last = nn.Conv2d(self.hidden_layer_dim[self.num_layers - 1], self.frame_channel , kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, input_tensor, mask_true):
        # Reshape frames and masks from [batch, length, height, width, channel]
        # to [batch, length, channel, height, width]
        # print("the dim of the input test img from the model " , input_tensor.shape)
        device = input_tensor.device

        frames = input_tensor.permute(0, 1, 4, 2, 3).contiguous()
        # frames = input_tensor
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        # mask_true = mask_true
        # print(mask_true.shape, frames.shape, "data dim")
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        # print("height",height,"width",width,"batch",batch)



        out_frames = []
        h_t = []
        c_t = []

        # Initialize hidden and cell states for all layers with zeros
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.hidden_layer_dim[i], height, width]).to(device)
            h_t.append(zeros)
            c_t.append(zeros)

        # Loop over each time step
        for t in range(self.total_frames_length - 1):
            if t < self.in_frames_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.in_frames_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.in_frames_length]) * x_gen

            # Forward pass through the first layer of ConvLSTM
            h_t[0], c_t[0] = self.cell_list[0](net, (h_t[0], c_t[0]))

            # Forward pass through the remaining layers of ConvLSTM
            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], (h_t[i], c_t[i]))

            # Generate the output for the current time step
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            out_frames.append(x_gen)

        # Stack the generated frames and reshape to [batch, length, height, width, channel]
        out_frames = torch.stack(out_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()

        loss = self.MSE_criterion(out_frames, input_tensor[:, 1:])

        return out_frames, loss