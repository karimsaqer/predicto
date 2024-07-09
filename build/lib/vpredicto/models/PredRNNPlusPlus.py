import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
from ..modules.predrnnpp.ghu import GHU
from ..modules.predrnnpp.causal_lstm import CausalLSTMCell

'''
PredRNN++ Model
__init__ method to initialize the PredRNN++ Model: you can pass the input_channels, hidden_channels, kernel_size, num_layers, and output_channels as parameters
forward method to pass the input and hidden state through the PredRNN++ Model
the input is the input and hidden state
the output is the predicted frames
'''
class PredRNNpp_Model(nn.Module):
    def __init__(self, input_channels = 1, hidden_channels = 64, kernel_size = 3, num_layers = 4, output_channels = 1):
        super(PredRNNpp_Model, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Initial convolution to get the hidden state
        self.initial_conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, padding=1)

        # Create the LSTM cells
        self.cells = nn.ModuleList([
            CausalLSTMCell(hidden_channels, hidden_channels, kernel_size)
            for _ in range(num_layers)
        ])
        # Create the GHU cell
        self.ghu = GHU(hidden_channels, hidden_channels, kernel_size)

        # Final convolution to get the output
        self.final_conv = nn.Conv2d(hidden_channels, output_channels, kernel_size=kernel_size, padding=1)

    def forward(self, x, pred_frames=10):
        batch_size, seq_len, _, height, width = x.size()

        # Initialize hidden, cell, and memory states
        h_t = [torch.zeros(batch_size, self.hidden_channels, height, width).to(x.device) for _ in range(self.num_layers)]
        c_t = [torch.zeros(batch_size, self.hidden_channels, height, width).to(x.device) for _ in range(self.num_layers)]
        m_t = torch.zeros(batch_size, self.hidden_channels, height, width).to(x.device)

        # Initialize z_t for the first layer
        z_t = None
        outputs = []

        # Encode the input sequence
        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            x_t = self.initial_conv(x_t)

            h_t[0], c_t[0], m_t = self.cells[0](x_t, h_t[0], c_t[0], m_t)
            for i in range(1, self.num_layers):
                if i == 1:
                    z_t = self.ghu(x_t, z_t)
                    x_t = z_t
                h_t[i], c_t[i], m_t = self.cells[i](h_t[i-1], h_t[i], c_t[i], m_t)

            outputs.append(h_t[-1].unsqueeze(1))

        # Predict future frames
        for t in range(pred_frames):
            x_t = self.final_conv(h_t[-1])  # Use the last hidden state to predict next frame
            x_t = self.initial_conv(x_t)

            h_t[0], c_t[0], m_t = self.cells[0](x_t, h_t[0], c_t[0], m_t)
            for i in range(1, self.num_layers):
                if i == 1:
                    z_t = self.ghu(x_t, z_t)
                    x_t = z_t
                h_t[i], c_t[i], m_t = self.cells[i](h_t[i-1], h_t[i], c_t[i], m_t)

            outputs.append(h_t[-1].unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        final_outputs = []

        # Apply final_conv to each time step separately
        for t in range(outputs.size(1)):
            final_outputs.append(self.final_conv(outputs[:, t, :, :, :]).unsqueeze(1))

        final_outputs = torch.cat(final_outputs, dim=1)
        return final_outputs[:, -pred_frames:]  # Return only the predicted frames

    def train_model(self, train_loader, lr=0.001, epochs=10, device='cpu'):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

        num_epochs = epochs
        pred_frames = 10  # Number of frames to predict after the initial sequence

        # Training loop
        for epoch in range(0, num_epochs):
            self.train()
            for batch in train_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = self(inputs, pred_frames=pred_frames)  # Include pred_frames argument
                loss = criterion(outputs, targets)  # Adjust loss calculation
                loss.backward()
                optimizer.step()

            # Print loss after each epoch
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


    def test_model(self, test_loader, device='cpu'):
        self.eval()
        total_loss = 0.0
        criterion = nn.MSELoss()
        self.to(device)

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)  
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Show sample predictions
                for i in range(3):
                    input_seq = inputs[i].cpu().numpy().squeeze()
                    target_seq = labels[i].cpu().numpy().squeeze()
                    output_seq = outputs[i].cpu().numpy().squeeze()

                    fig, axes = plt.subplots(3, input_seq.shape[0], figsize=(15, 5))
                    for t in range(input_seq.shape[0]):
                        axes[0, t].imshow(input_seq[t], cmap='gray')
                        axes[0, t].axis('off')
                        axes[0, t].set_title(f'Input {t+1}')

                        axes[1, t].imshow(output_seq[t], cmap='gray')
                        axes[1, t].axis('off')
                        axes[1, t].set_title(f'Output {t+1}')

                        axes[2, t].imshow(target_seq[t], cmap='gray')
                        axes[2, t].axis('off')
                        axes[2, t].set_title(f'Target {t+1}')

                    plt.show()

                break  # Only visualize for the first batch

        print(f"Test Loss: {total_loss / len(test_loader)}")

    def evaluate_model(model, test_loader, criterion, pred_frames,device='cpu'):
        model.eval()
        total_loss = 0
        ssim_scores = []
        psnr_scores = []
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs, pred_frames=pred_frames)
                loss = criterion(outputs[:, -pred_frames:], targets)
                total_loss += loss.item()

                for i in range(outputs.size(0)):
                    output = outputs[i, -pred_frames:].cpu().numpy().squeeze()
                    target = targets[i].cpu().numpy().squeeze()

                    # Compute SSIM and PSNR
                    ssim_scores.append(ssim(output, target, data_range=target.max() - target.min()))
                    psnr_scores.append(psnr(output, target, data_range=target.max() - target.min()))

        avg_loss = total_loss / len(test_loader)
        avg_ssim = np.mean(ssim_scores)
        avg_psnr = np.mean(psnr_scores)

        # print(f'Test Loss: {avg_loss:.4f}, SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}')
        return avg_loss, avg_ssim, avg_psnr

    def evaluate_ssim(self, test_loader, device='cpu'):
        _, ssim, __ = self.evaluate_model(self, test_loader, nn.MSELoss(), 10, device)
        print(f'Average SSIM: {ssim:.4f}')

    def evaluate_MSE(self, test_loader, device='cpu'):
        mse, _, __ = self.evaluate_model(self, test_loader, nn.MSELoss(), 10, device)
        print(f'Average MSE: {mse:.4f}')

    def evaluate_PSNR(self, test_loader, device='cpu'):
        __, __, psnr = self.evaluate_model(self, test_loader, nn.MSELoss(), 10, device)
        print(f'Average PSNR: {psnr:.4f}')