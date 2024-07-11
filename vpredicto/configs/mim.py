class configs():
  def __init__(self):
        self.configs={
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
            'device':"cuda",
            'model_name':"mim"
            }
  
  def print_dict(self):
      print(self.configs)
  def update_dict(self , key , value):
    self.configs[key]=value

