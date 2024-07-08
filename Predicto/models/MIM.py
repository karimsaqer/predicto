import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from ..Utils.utils import show_video_line, patch_images, patch_images_back, schedule_sampling
from ..modules.MIM import MIM_Model




class MIMLightningModel(pl.LightningModule):
    def __init__(self , configs = {
      'in_shape': [10, 1, 64, 64],
      'filter_size': 5,
      'stride': 1,
      'patch_size':2,
      'in_frames_length': 10,
      'out_frames_length': 10,
      "total_length":20,
      "batch_size" : 4,
      "r_sampling_step_1" : 25000,
      "r_sampling_step_2" : 50000,
      "r_exp_alpha" : 5000,
      "sampling_stop_iter" : 50000,
      "sampling_start_value" : 1.0,
      "sampling_changing_rate" : 0.00002,
      "num_hidden":[128,128,128,128],
      "lr":0.001,
      "layer_norm":0,
      'device':"cuda"
      }):
        super(MIMLightningModel, self).__init__()
        self.configs=configs
        self.model = MIM_Model(configs)
        self.lr = configs["lr"]
        self.eta=1.0
        self.criterion = nn.MSELoss()


    def forward(self, x, mask_true):

        #print("from the forward model",x.shape,"shape of x ")

        mask_input = self.configs["in_frames_length"]
        _, channels, height, width = self.configs["in_shape"]

        # preprocess
        test_ims = torch.cat([x, mask_true], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        test_dat = patch_images(test_ims, self.configs["patch_size"]) # frames tensor
        test_ims = test_ims[:, :, :, :, :channels]
        # print(channels)
        real_input_flag = torch.zeros(
            (x.shape[0],
            self.configs["total_length"] - mask_input - 1,
            height // self.configs["patch_size"],
            width // self.configs["patch_size"],
            self.configs["patch_size"] ** 2 * channels)).to(self.device)

        new_frames, _ = self.model(test_dat , real_input_flag)
        # print(new_frames.shape)

        new_frames = patch_images_back(new_frames, self.configs["patch_size"])

        pred_frames = new_frames[:, -self.configs["out_frames_length"]:].permute(0, 1, 4, 2, 3).contiguous()

        return pred_frames

    def training_step(self, batch, batch_idx):
        frames, mask_true = batch
        test_ims = torch.cat([frames, mask_true], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        test_dat = patch_images(test_ims, self.configs["patch_size"]) # frames tensor
        self.eta, real_input = schedule_sampling( test_dat.shape[0], self.eta, self.global_step, self.configs)

        new_frames , loss = self.model(test_dat, real_input)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)

        return loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_model(self, train_loader, lr=0.001, epochs=10, device='cpu'):
        self.to(self.configs["device"])
        self.lr = lr
        logger = TensorBoardLogger('tb_logs', name='MIM_1')
        trainer = Trainer(max_epochs=epochs , logger=logger, accelerator=self.configs["device"])
        trainer.fit(self, train_loader)

    def test_model(self, test_loader, device='cpu'):
        self.to(device)
        self.to(device)
        self.eval()
        pre_seq_length = self.configs['in_frames_length']
        aft_seq_length = self.configs['out_frames_length']
        for batch in test_loader:
            frames, mask_true = batch
            frames = frames.to(device)
            mask_true = mask_true.to(device)

            print('>' * 35 + ' Input ' + '<' * 35)
            show_video_line(frames[0].cpu().numpy(), ncols=pre_seq_length, vmax=0.6, cbar=False, out_path=None, format='png', use_rgb=True)

            print('>' * 35 + ' True Output ' + '<' * 35)
            show_video_line(mask_true[0].cpu().numpy(), ncols=aft_seq_length, vmax=0.6, cbar=False, out_path=None, format='png', use_rgb=True)

            with torch.no_grad():
                pred_y = self(frames, mask_true)

            print('>' * 35 + ' Predicted Output ' + '<' * 35)
            show_video_line(pred_y[0].cpu().numpy(), ncols=aft_seq_length, vmax=0.6, cbar=False, out_path=None, format='png', use_rgb=False)
            break