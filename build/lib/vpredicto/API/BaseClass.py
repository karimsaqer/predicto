import torch
from ..models.predrnnpp import PredRNNpp_Model

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

        self.model = model.to(self.device) if model else PredRNNpp_Model().to(self.device)

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
    def Predict(self, test_loader):
        self.model.test_model(test_loader, self.device)


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
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")