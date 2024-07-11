from ..modules.gan.Generator import Generator
from ..modules.gan.Discriminator import Discriminator
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import os

"""
Generative Adversarial Network (GAN) Model for Video Prediction.
"""

class GANModel:
    def __init__(self, input_frames_num=10, pred_frames_num=10, n_residual_blocks=9, lr=0.001, device='cuda',betas=(0.5, 0.999)):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.generator = Generator(input_frames_num, pred_frames_num).to(self.device)
        self.frame_discriminator = Discriminator(1).to(self.device)
        self.sequence_discriminator = Discriminator(pred_frames_num).to(self.device)
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.optimizer_DA = optim.Adam(self.frame_discriminator.parameters(), lr=lr, betas=betas)
        self.optimizer_DB = optim.Adam(self.sequence_discriminator.parameters(), lr=lr, betas=betas)
        self.adversarial_loss = nn.MSELoss()
        self.reconstruction_loss = self.l1_loss
        self.input_frames_num = input_frames_num
        self.pred_frames_num = pred_frames_num

        # In the init method, print the parameters of the model
        print("Model Parameters")
        print("input_frames_num: ", self.input_frames_num)
        print("pred_frames_num: ", self.pred_frames_num)
        print("n_residual_blocks: ", n_residual_blocks)
        print("lr: ", lr)
        print("device: ", self.device)
        print("betas: ", betas)


    """
    L1 Loss: Mean Absolute Error between two tensors.
    Used for reconstruction loss in the generator.
    """
    @staticmethod
    def l1_loss(x, y):
        return torch.mean(torch.abs(x - y))

    """
    Laplacian of Gaussian (LoG): Applies LoG filter to input tensor.

    Purpose:
    - Enhance edges and suppress noise.

    Inputs:
    - x: Input tensor.

    Outputs:
    - output: Tensor with LoG applied to each channel.
    """
    @staticmethod
    def laplacian_of_gaussian(x):
        device = x.device
        batch_size, channels, height, width = x.shape
        laplacian_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(device)
        laplacian_filter.weight = nn.Parameter(torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).reshape(1, 1, 3, 3).to(device))
        output = torch.zeros_like(x)
        for c in range(channels):
            output[:, c:c+1, :, :] = laplacian_filter(x[:, c:c+1, :, :])
        return output

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
        self.generator.eval()
        total_loss = 0
        ssim_scores = []
        psnr_scores = []
        mse_loss = 0
        with torch.no_grad():
            for batch in loader:
                video = batch.to(self.device).float().squeeze(2)
                inputs = video[:, :10, :, :]  # 10 frames for input
                targets = video[:, 10:20, :, :]  # 10 frames for target
                outputs = self.generator(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                mse_loss += nn.functional.mse_loss(outputs, targets).item()
                for i in range(outputs.size(0)):
                    output = outputs[i].cpu().numpy().squeeze()
                    target = targets[i].cpu().numpy().squeeze()
                    ssim_scores.append(ssim(output, target, data_range=target.max() - target.min()))
                    psnr_scores.append(psnr(output, target, data_range=target.max() - target.min()))
        avg_loss = total_loss / len(loader)
        avg_ssim = np.mean(ssim_scores)
        avg_psnr = np.mean(psnr_scores)
        mse_loss = mse_loss / len(loader)
        return avg_loss, avg_ssim, avg_psnr, mse_loss

    """
    Trains the model on a given dataset loader.
    inputs:
    - train_loader: DataLoader for the training dataset.
    - lr: Learning rate for the optimizer.
    - epochs: Number of epochs to train the model.
    - device: Device to train the model on (cpu or cuda).
    """
    def train_model(self, train_loader, lr=0.001, epochs=10, device='cuda'):
        self.to(self.device)

        for epoch in range(epochs):
            print(f"Training Epoch [{epoch}/{epochs}]")
            for i, video in enumerate(train_loader):
                video = video.to(self.device).float().squeeze(2)
                # Prepare inputs
                input_frames = video[:, :self.input_frames_num, :, :]
                target_frames = video[:, self.input_frames_num:self.input_frames_num+self.pred_frames_num, :, :]

                # Train Generators
                self.optimizer_G.zero_grad()

                # Generate fake frames
                fake_frames = self.generator(input_frames)
                # Calculate reconstruction loss (L1 loss) between fake and real frames  
                loss_image = self.reconstruction_loss(fake_frames, target_frames)

                # Calculate Laplacian of Gaussian (LoG) loss to enhance edge consistency
                loss_log = self.reconstruction_loss(self.laplacian_of_gaussian(fake_frames), self.laplacian_of_gaussian(target_frames))

                # Discriminator A (Frame Discriminator) loss
                pred_fake = self.frame_discriminator(fake_frames.view(-1, 1, fake_frames.size(2), fake_frames.size(3)))
                loss_G_adv_frame = self.adversarial_loss(pred_fake, torch.ones_like(pred_fake))
                
                # Discriminator B (Sequence Discriminator) loss
                pred_fake_seq = self.sequence_discriminator(fake_frames)
                loss_G_adv_seq = self.adversarial_loss(pred_fake_seq, torch.ones_like(pred_fake_seq))

                # Total generator loss
                loss_G = loss_image + loss_log + loss_G_adv_frame + loss_G_adv_seq
                loss_G.backward()
                self.optimizer_G.step()

                # Train Frame Discriminator (Discriminator A)
                self.optimizer_DA.zero_grad()

                # Real frames loss
                real_frames = target_frames.view(-1, 1, target_frames.size(2), target_frames.size(3))
                pred_real = self.frame_discriminator(real_frames)
                loss_DA_real = self.adversarial_loss(pred_real, torch.ones_like(pred_real))

                # Fake frames loss
                pred_fake = self.frame_discriminator(fake_frames.view(-1, 1, fake_frames.size(2), fake_frames.size(3)).detach())
                loss_DA_fake = self.adversarial_loss(pred_fake, torch.zeros_like(pred_fake))

                # Total discriminator A loss
                loss_DA = (loss_DA_real + loss_DA_fake) * 0.5
                loss_DA.backward()
                self.optimizer_DA.step()
                
                # Train Sequence Discriminator (Discriminator B)
                self.optimizer_DB.zero_grad()

                # Real sequence loss
                pred_real_seq = self.sequence_discriminator(target_frames)
                loss_DB_real = self.adversarial_loss(pred_real_seq, torch.ones_like(pred_real_seq))

                # Fake sequence loss
                pred_fake_seq = self.sequence_discriminator(fake_frames.detach())
                loss_DB_fake = self.adversarial_loss(pred_fake_seq, torch.zeros_like(pred_fake_seq))

                # Total discriminator B loss
                loss_DB = (loss_DB_real + loss_DB_fake) * 0.5
                loss_DB.backward()
                self.optimizer_DB.step()
                if i % 50 == 0:
                  print(f"Loss_G: {loss_G.item():.4f}, Loss_DA: {loss_DA.item():.4f}, Loss_DB: {loss_DB.item():.4f}")


    """
    Test the model on a given dataset loader.
    inputs:
    - test_loader: DataLoader for the
    """
    def test_model(self, test_loader, device='cuda', save=True):
        self.generator.eval()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        save_dir = 'pred_results'
        save = True
        if save and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        all_output_frames = []

        for batch_idx, sample_batch in enumerate(test_loader):
            input_frames = sample_batch[:, :10, :, :].float().squeeze(2).to(self.device)
            target_frames = sample_batch[:, 10:20, :, :].float().squeeze(2).to(self.device)

            with torch.no_grad():
                output_frames = self.generator(input_frames)
                all_output_frames.append(output_frames.cpu())

            if save:
                # Save each frame in the batch as an image
                for frame_idx in range(output_frames.size(0)):
                    fig, axs = plt.subplots(3, 10, figsize=(20, 6))
                    for i in range(10):
                        axs[0, i].imshow(input_frames[frame_idx, i].cpu().numpy(), cmap='gray')
                        axs[0, i].axis('off')
                        axs[0, i].set_title(f'Input Frame {i}')
                    for i in range(10):
                        axs[1, i].imshow(target_frames[frame_idx, i].cpu().numpy(), cmap='gray')
                        axs[1, i].axis('off')
                        axs[1, i].set_title(f'Target Frame {i}')
                    for i in range(10):
                        axs[2, i].imshow(output_frames[frame_idx, i].cpu().numpy(), cmap='gray')
                        axs[2, i].axis('off')
                        axs[2, i].set_title(f'Generated Frame {i}')
                    plt.tight_layout()

                    save_path = os.path.join(save_dir, f'batch_{batch_idx}_frame_{frame_idx}.png')
                    plt.savefig(save_path)
                    plt.close(fig)

        # Concatenate all batches into one tensor
        all_output_frames = torch.cat(all_output_frames, dim=0)

        return all_output_frames


    """
    Evaluate the model on a given dataset loader.
    """
    def evaluate_ssim(self, test_loader, device='cpu'):
        _, ssim, __, __ = self.evaluate_model(test_loader, nn.MSELoss(), 10)
        print(f'Average SSIM: {ssim:.4f}')

    def evaluate_MSE(self, test_loader, device='cpu'):
        __, __, __, mse = self.evaluate_model(test_loader, nn.MSELoss(), 10)
        print(f'Average MSE: {mse:.4f}')

    def evaluate_PSNR(self, test_loader, device='cpu'):
        __, __, psnr, __ = self.evaluate_model(test_loader, nn.MSELoss(), 10)
        print(f'Average PSNR: {psnr:.4f}')

    def to(self, device):
        self.device = device
        self.generator.to(device)
        self.frame_discriminator.to(device)
        self.sequence_discriminator.to(device)
        return self