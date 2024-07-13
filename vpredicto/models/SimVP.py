from ..modules.simvp.Encoder import Encoder
from ..modules.simvp.Translator import Translator
from ..modules.simvp.Decoder import Decoder
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

class SimVP(nn.Module):
    def __init__(self, shape_in=(10, 1, 64, 64), hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3, 5, 7, 11], groups=8,device='cuda'):
        super(SimVP, self).__init__()
        '''
        shape_in == Input shape as a tuple (T, C, H, W)
        hid_S == Number of hidden channels in the spatial encoder/decoder
        hid_T == Number of hidden channels in the temporal translator
        N_S == Number of convolutional layers in the encoder/decoder
        N_T == Number of Inception blocks in the translator
        incep_ker == List of kernel sizes for Inception blocks
        groups == Number of groups to separate the channels in GroupNorm
        '''
        T, C, H, W = shape_in

        # Initialize encoder
        self.encoder = Encoder(C, hid_S, N_S)

        # Initialize translator
        self.translator = Translator(T * hid_S, hid_T, N_T, incep_ker, groups)

        # Initialize decoder
        self.decoder = Decoder(hid_S, C, N_S)

        if device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                print("CUDA is not available, CPU will be used")
        else:
            self.device = torch.device("cpu")

        # In the __init__ method, print the parameters of the model
        print('The parameters of the model are: \n')
        print(f'shape_in: {shape_in}, hid_S: {hid_S}, hid_T: {hid_T}, N_S: {N_S}, N_T: {N_T}, incep_ker: {incep_ker}, groups: {groups}')

    def forward(self, x_raw):
        '''
        Forward pass through the SimVP model.
        x_raw: Input tensor of shape (B, T, C, H, W)
        B: Batch size
        T: Temporal dimension (number of frames)
        C: Number of input channels
        H: Height of input
        W: Width of input
        '''
        B, T, C, H, W = x_raw.shape

        # Reshape input to (B * T, C, H, W) for processing through the encoder
        x = x_raw.view(B * T, C, H, W)

        # Encode input
        encoded, skip_connection = self.encoder(x)

        # Reshape encoded output to (B, T, hidden_channels, H, W) for the translator
        encoded = encoded.view(B, T, encoded.shape[1], encoded.shape[2], encoded.shape[3])

        # Translate encoded output
        translated = self.translator(encoded)

        # Reshape translated output back to (B * T, hidden_channels, H, W) for the decoder
        translated = translated.view(B * T, encoded.shape[2], encoded.shape[3], encoded.shape[4])

        # Decode translated output
        decoded = self.decoder(translated, skip_connection)

        # Reshape decoded output back to (B, T, C, H, W)
        output = decoded.view(B, T, C, H, W)

        return output

    def train_model(self, train_loader, lr=0.001, epochs=10, device='cuda'):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0
            train_loader = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training')
            for data in train_loader:
                data = data.to(device).float()
                input = data[:, :10, :, :, :].to(device)
                target = data[:, 10:, :, :, :].to(device)
                optimizer.zero_grad()
                output = self(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                train_loader.set_postfix(loss=loss.item())
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

    def test_model(self, test_loader, device='cuda', save=True):
        self.to(device)
        self.eval()

        save_dir = 'output_frames'
        if save and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        all_output_frames = []

        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                data = data.to(device).float()
                input = data[:, :10, :, :, :]
                target = data[:, 10:, :, :, :]
                output = self(input)
                input = input.detach().cpu().numpy()
                target = target.detach().cpu().numpy()
                output = output.detach().cpu().numpy()
                all_output_frames.append(output)

                if save:
                    self.plot_frames(input[0], target[0], output[0], frame_idx=range(0, 10), save_dir=save_dir, batch_idx=batch_idx)

                break  # Only visualize for the first batch

        return torch.cat([torch.tensor(f) for f in all_output_frames], dim=0)

    def plot_frames(self, input_frames, target_frames, output_frames, frame_idx, save_dir, batch_idx):
        num_frames = len(frame_idx)
        fig, axes = plt.subplots(3, num_frames, figsize=(15, 5))

        for i, idx in enumerate(frame_idx):
            axes[0, i].imshow(input_frames[idx].transpose(1, 2, 0), cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Input {idx+1}')

            axes[1, i].imshow(target_frames[idx].transpose(1, 2, 0), cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f'Target {idx+1}')

            axes[2, i].imshow(output_frames[idx].transpose(1, 2, 0), cmap='gray')
            axes[2, i].axis('off')
            axes[2, i].set_title(f'Output {idx+1}')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'batch_{batch_idx}.png'))
        plt.show()
    
    """
    Evaluates the model on a given dataset loader.

    Inputs:
    - model: The trained model to evaluate.
    - loader: DataLoader for the dataset.
    - criterion: Loss function.
    - pred_frames_num: Number of frames to predict.

    Outputs:
    - avg_loss: Average loss over the dataset.
    - avg_ssim: Average Structural Similarity Index (SSIM) over the dataset.
    - avg_psnr: Average Peak Signal-to-Noise Ratio (PSNR) over the dataset.
    - mse_loss: Mean Squared Error (MSE) over the dataset.
    """
    def evaluate_model(self, loader, criterion, pred_frames):
        self.eval()
        total_loss = 0
        ssim_scores = []
        psnr_scores = []
        mse_loss = 0

        self.to(self.device)  # Ensure model is on the correct device

        with torch.no_grad():
            for batch in loader:
                video = batch.to(self.device).float()  # Ensure data is on the correct device
                inputs = video[:, :10, :, :, :]  # 10 frames for input
                targets = video[:, 10:20, :, :, :]  # 10 frames for target

                outputs = self(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                mse_loss += nn.functional.mse_loss(outputs, targets).item()

                for i in range(outputs.size(0)):
                    output = outputs[i].cpu().numpy().squeeze()
                    target = targets[i].cpu().numpy().squeeze()
                    ssim_scores.append(ssim(output, target, data_range=target.max() - target.min(), multichannel=True))
                    psnr_scores.append(psnr(output, target, data_range=target.max() - target.min()))

        avg_loss = total_loss / len(loader)
        avg_ssim = np.mean(ssim_scores)
        avg_psnr = np.mean(psnr_scores)
        mse_loss = mse_loss / len(loader)

        # return {"loss": avg_loss, "ssim": avg_ssim, "psnr": avg_psnr, "mse": mse_loss}
        return avg_loss, avg_ssim, avg_psnr, mse_loss
    
    """
    Evaluate the model on a given dataset loader.
    """
    def evaluate_ssim(self, test_loader, device='cuda'):
        _, avg_ssim, _, _ = self.evaluate_model(test_loader, nn.MSELoss(), 10)
        print(f'Average SSIM: {avg_ssim:.4f}')


    def evaluate_MSE(self, test_loader, device='cuda'):
        _, _, _, mse_loss = self.evaluate_model(test_loader, nn.MSELoss(), 10)
        print(f'Average MSE: {mse_loss:.4f}')

    def evaluate_PSNR(self, test_loader, device='cuda'):
        _, _, avg_psnr, _ = self.evaluate_model(test_loader, nn.MSELoss(), 10)
        print(f'Average PSNR: {avg_psnr:.4f}')