import torch
from ..models.ConvLSTM import ConvLSTMModule
from ..models.MIM import MIMLightningModel
from ..models.PredRNNPlusPlus import PredRNNpp_Model



# Adding configs 

# Base class for all models
class Predicto:
    '''
    init method to initialize the model and device: you can pass the model that you chose and device as parameters
    '''
    def __init__(self, configs = None):
        if configs==None or configs=='':
            configs={
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
                }
        if configs["device"].lower() == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                print("CUDA is not available, CPU will be used")
        else:
            self.device = torch.device("cpu")

        # if 
        # self.model = model.to(self.device) if model else ConvLSTMModule().to(self.device)
        self.device = configs["device"] if configs else "cpu"
        if configs['model_name']==None:
          self.model=ConvLSTMModule(configs)
        elif configs['model_name'].lower()=='mim':
          self.model=MIMLightningModel(configs)
        elif configs['model_name'].lower()=='convlstm':
          self.model=ConvLSTMModule(configs)
        elif configs['model_name'].lower()=='predrnn++':
          self.model=PredRNNpp_Model(configs)
        elif configs['model_name'].lower()=='simvp':
          self.model=ConvLSTMModule(configs)
        elif configs['model_name'].lower()=='prednet':
          self.model=ConvLSTMModule(configs)
        elif configs['model_name'].lower()=='gan':
          self.model=ConvLSTMModule(configs)

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


    '''
    load_pkl method to load the model from a .pkl file: you can pass the path as a parameter
    the input is the path to the .pkl file
    the output is the loaded model
    '''
    def load_pkl(self, pkl_file_path='model.pth'):
        state_dict = torch.load(pkl_file_path)
        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {pkl_file_path}")