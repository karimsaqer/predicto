import torch
from ..models.ConvLSTM import ConvLSTMModule
from ..models.MIM import MIMLightningModel
from ..models.PredRNNPlusPlus import PredRNNpp_Model
from ..models.SimVP import SimVP
from ..models.GAN import GANModel
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MovingMNIST

# Base class for all models
class Predicto:
    '''
    init method to initialize the model and device: you can pass the model that you chose and device as parameters
    '''
    def __init__(self, model=None, device='cpu'):
        if device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                print("CUDA is not available, CPU will be used")
        else:
            self.device = torch.device("cpu")


        self.model = model.to(self.device) if model else ConvLSTMModule().to(self.device)

    '''
    train method to train the model: you can pass the train_loader, learning rate, and number of epochs as parameters
    the input is the train_loader, learning rate, and number of epochs
    the output is the trained model
    '''
    def train(self, train_loader, lr=0.001, epochs=10):
        self.model.train_model(train_loader, lr, epochs, self.device)

    '''
    predict method to test the model: you can pass the test_loader as a parameter
    the input is the test_loader
    the output is the output of test data from the model
    '''
    def Predict(self, test_loader, save=True):
        self.model.test_model(test_loader, self.device, save=save)


    '''
    evaluate method to evaluate the model: you can pass the test_loader as a parameter
    the input is the test_loader
    the output is the evaluation of the model
    '''
    # We can do it in the base class (same logic)
    def evaluate(self, test_loader, SSIM=False, MSE=True, PSNR= False): # Here Adding any new evaluation metric
        if SSIM:
            self.model.evaluate_ssim(test_loader, self.device)
        if MSE:
            self.model.evaluate_MSE(test_loader, self.device)
        if PSNR:
            self.model.evaluate_PSNR(test_loader, self.device)
        # else:
        #     self.model.test_model(test_loader, self.device)

    '''
    save method to save the model: you can pass the path as a parameter
    the input is the path
    the output is the saved model
    '''
    def save(self, path='model.pth'):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    '''
    load method to load the model: you can pass the path as a parameter
    the input is the path
    the output is the saved model
    '''
    def load(self, path='model.pth'):
        if isinstance(self.model, GANModel):
            self.model.generator.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Generator model loaded from {path}")
        else:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Model loaded from {path}")


    '''
    load_pkl method to load the model from a .pkl file: you can pass the path as a parameter
    the input is the path to the .pkl file
    the output is the loaded model
    '''
    def load_pkl(self, pkl_file_path='model.pkl'):
        state_dict = torch.load(pkl_file_path)
        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {pkl_file_path}")


    '''
    get_data_loaders method to get the train and test data loaders: you can pass the batch size and train size as parameters
    the input is the batch size and train size
    the output is the train and test
    '''
    def get_data_loaders(batch_size=4, train_size=0.9):
      dataset = MovingMNIST(root='data/', download=True)
      num_samples = len(dataset)
      train_size = int(train_size * num_samples)
      test_size = num_samples - train_size
      input_frames = 10
      predicted_frames = 10
      train_dataset, test_dataset = random_split(dataset[:num_samples], [train_size, test_size])
      x_train, y_train = split_dataset(train_dataset, input_frames, predicted_frames)
      x_test, y_test = split_dataset(test_dataset, input_frames, predicted_frames)
      train_loader = DataLoader(list(zip(x_train, y_train)), batch_size=batch_size, shuffle=True)
      test_loader = DataLoader(list(zip(x_test, y_test)), batch_size=batch_size, shuffle=False)

      return train_loader, test_loader

# Function to split sequence into X and Y
def split_dataset(dataset, input_frames, predicted_frames):
    X, Y = [], []
    for sequence in dataset:
        for i in range(len(sequence) - input_frames - predicted_frames + 1):
            X.append(sequence[i:i+input_frames].float())
            Y.append(sequence[i+input_frames:i+input_frames+predicted_frames].float())
    return torch.stack(X), torch.stack(Y)