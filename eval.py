from ImageDataset import * 
from torch import nn
from main import UNet


import torch
import numpy as np

# +++++ ===== +++++ ===== +++++ ===== +++++ ===== 

test_path = "dataset/test"
eval_path = "dataset/eval"
batch_size = 32
num_models = 5
# +++++ ===== +++++ ===== +++++ ===== +++++ ===== 


def main():
    #define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model 
    model = UNet(input_channels=3, output_channels=2)
    criterion = nn.MSELoss()

    # load image datasets
    test_loader = load_dataset(images_path=test_path, batch_size=batch_size)
    eval_loader = load_dataset(images_path=eval_path, batch_size=batch_size)


    total_loss = 0
    # evaluate
    for batch in eval_loader:
        images = batch['images'].to(device)
        masked_images = batch['masked_images'].to(device)

        # calculate tloss
        output = model(masked_images)
        loss = criterion(output, images[:, :2, :, :])
        total_loss += sum(loss)
    print(total_loss)