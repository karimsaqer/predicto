# utils.py

import math
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

def show_video_line(data, ncols, vmax=0.6, vmin=0.0, cmap='gray', norm=None, cbar=False, format='png', out_path=None, use_rgb=False):
    """generate images with a video sequence"""
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(3.25 * ncols, 3))
    plt.subplots_adjust(wspace=0.01, hspace=0)

    if len(data.shape) > 3:
        data = data.swapaxes(1,2).swapaxes(2,3)

    images = []
    if ncols == 1:
        if use_rgb:
            im = axes.imshow(cv2.cvtColor(data[0], cv2.COLOR_BGR2RGB))
        else:
            im = axes.imshow(data[0], cmap=cmap, norm=norm)
        images.append(im)
        axes.axis('off')
        im.set_clim(vmin, vmax)
    else:
        for t, ax in enumerate(axes.flat):
            if use_rgb:
                im = ax.imshow(cv2.cvtColor(data[t], cv2.COLOR_BGR2RGB), cmap='gray')
            else:
                im = ax.imshow(data[t], cmap=cmap, norm=norm)
            images.append(im)
            ax.axis('off')
            im.set_clim(vmin, vmax)

    if cbar and ncols > 1:
        cbaxes = fig.add_axes([0.9, 0.15, 0.04 / ncols, 0.7])
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.1, cax=cbaxes)

    plt.show()
    if out_path is not None:
        fig.savefig(out_path, format=format, pad_inches=0, bbox_inches='tight')
    plt.close()

def patch_images(input_tensor , patch_size):
    # Ensure the input tensor has 5 dimensions
    assert 5 == input_tensor.ndim

    # Extract the shape of the input tensor
    batch_size, frames_length, height, width, channels_num = input_tensor.shape

    # Reshape the input tensor to create patches
    input_tensor_reshaped = input_tensor.reshape(batch_size, frames_length,
                                                 height // patch_size, patch_size,
                                                 width // patch_size, patch_size,
                                                 channels_num)

    # Transpose the tensor to reorder dimensions for patching
    input_tensor_reshaped_transpose = input_tensor_reshaped.transpose(3, 4)

    # Reshape the transposed tensor to get the final patched tensor
    patched_input_tensor = input_tensor_reshaped_transpose.reshape(batch_size, frames_length,
                                                                   height // patch_size,
                                                                   width // patch_size,
                                                                   patch_size * patch_size * channels_num)

    # Return the patched tensor
    return patched_input_tensor

# Function to revert patched images back to their original form
def patch_images_back(output_tensor , patch_size):
    # Extract the shape of the output tensor
    batch_size, frames_length, height, width, channels_num = output_tensor.shape

    # Calculate the original number of channels
    channels = channels_num // (patch_size * patch_size)

    # Reshape the output tensor to revert patches
    output_tensor_reshaped = output_tensor.reshape(batch_size, frames_length,
                                                   height, width,
                                                   patch_size, patch_size,
                                                   channels)

    # Transpose the tensor to reorder dimensions back to original
    output_tensor_transposed = output_tensor_reshaped.transpose(3, 4)

    # Reshape the transposed tensor to get the final image tensor
    out_img = output_tensor_transposed.reshape(batch_size, frames_length,
                                               height * patch_size,
                                               width * patch_size,
                                               channels)

    # Return the final image tensor
    return out_img

def schedule_sampling(batch_size, eta, itr, args):
    # Extract shape parameters from the arguments
    T, channels_num, height, width = args["in_shape"]

    # Initialize a tensor of zeros for the input flag
    zeros = np.zeros((batch_size,
                      args["out_frames_length"]-1,
                      height // args["patch_size"],
                      width // args["patch_size"],
                      args["patch_size"] * args["patch_size"] * channels_num))


    # Update eta based on the iteration and sampling stop iteration
    if itr < args["sampling_stop_iter"]:
        eta -= args["sampling_changing_rate"]
    else:
        eta = 0.0

    # Generate random samples for determining true tokens
    random_sample_flipping = np.random.random_sample(
        (batch_size, args["out_frames_length"] - 1))

    # Determine which tokens are true based on eta
    true_token = (random_sample_flipping < eta)

    # Create tensors of ones and zeros for true and false tokens
    ones = np.ones((height // args["patch_size"],
                    width // args["patch_size"],
                    args["patch_size"] * args["patch_size"] * channels_num))
    zeros = np.zeros((height // args["patch_size"],
                      width // args["patch_size"],
                      args["patch_size"] * args["patch_size"] * channels_num))

    # Initialize a list to hold the input flag
    input_flag = []

    # Populate the input flag based on true tokens
    for i in range(batch_size):
        for j in range(args["out_frames_length"] - 1):
            if true_token[i, j]:
                input_flag.append(ones)
            else:
                input_flag.append(zeros)

    # Convert the input flag list to a numpy array
    input_flag = np.array(input_flag)

    # Reshape the input flag array to the required dimensions
    input_flag = np.reshape(input_flag,
                            (batch_size,
                             args["out_frames_length"] - 1,
                             height // args["patch_size"],
                             width // args["patch_size"],
                             args["patch_size"] * args["patch_size"] * channels_num))

    # Convert the input flag to a torch FloatTensor and move it to the specified device
    return eta, torch.FloatTensor(input_flag).to(args["device"])

class MovingMNIST(Dataset):
    def __init__(self, root,file_name ,
                 input_frames=10, output_frames=10,train_data=True,test_data=False):
        super(MovingMNIST, self).__init__()
        # the number of the input frames
        self.input_frames = input_frames
        #the number of the output frames
        self.output_frames = output_frames
         # load the file from the root dir
        self.dataset = np.expand_dims(
            np.load(os.path.join(root, file_name)),
            axis=-1
        )
        # check if the data for training or not to split the data

        if train_data:
            self.dataset = self.dataset[:,:8000]
        elif test_data:
            self.dataset = self.dataset[:,8000:9000]
        else:
            self.dataset = self.dataset[:,9000:]








    def change_torch(self,data_images_in , data_images_out):
      return torch.from_numpy(data_images_in / 255.0).contiguous().float() , torch.from_numpy(data_images_out / 255.0).contiguous().float()



    def __getitem__(self, idx):

        # get the item by the index
        frames_data = self.dataset[:, idx, ...]
        # transpose the item
        frames_data = frames_data.transpose(0, 3, 1, 2)

        # get the total lenght of the frames input and output
        total_lenght = self.input_frames + self.output_frames

        # split the data to the in and out data frames
        in_data = frames_data[:self.input_frames]
        out_data = frames_data[self.input_frames:total_lenght]

        return self.change_torch(in_data , out_data)

    def __len__(self):
        return self.dataset.shape[1]


def get_dataset( data_root_dir , file_name, train_batch_size, val_batch_size , test_batch_size , num_workers=4,
              in_length=10, out_length=10):

    # get the training dataset
    dataset_training = MovingMNIST(root=data_root_dir,file_name=file_name, train_data=True,test_data=False,
                            input_frames=in_length,
                            output_frames=out_length)

    # get the test dataset
    dataset_test = MovingMNIST(root=data_root_dir,file_name=file_name, train_data=False,test_data=True,
                            input_frames=in_length,
                            output_frames=out_length)
    # get the Validation dataset
    dataset_validation = MovingMNIST(root=data_root_dir,file_name=file_name, train_data=False,test_data=False,
                            input_frames=in_length,
                            output_frames=out_length)

    dataloader_train =  DataLoader(
            dataset= dataset_training,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers
        )


    dataloader_validation = DataLoader(
            dataset= dataset_validation,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers
        )


    dataloader_test =  DataLoader(
            dataset= dataset_test,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    return dataloader_train, dataloader_validation, dataloader_test