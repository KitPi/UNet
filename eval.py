from ImageDataset import * 
from torchvision import transforms

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

    # load image datasets
    test_loader = load_dataset(images_path=test_path, batch_size=batch_size)
    eval_loader = load_dataset(images_path=eval_path, batch_size=batch_size)

    # 