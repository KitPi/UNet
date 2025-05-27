from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from skimage import transforms
from PIL import Image

import os
import torch 
import numpy as np

from main import UNet

# load models
model = UNet(input_channels=3, output_channels=2)

## parameters
batch_size = 16
train_folder = "dataset/train/"
test_folder = "dataset/test/"
val_folder = "dataset/val/"
output_dir = "output/"


# dataset
class ImageDataset(Dataset):
    def __init__(self, images_path, transform = None):
        self.image_list = [os.path.join(images_path, im) for im in os.listdir(images_path)] # Image list
        self.transform = transform
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, index):
        image_path = self.image_list[index]
        # attempt to open images and convert to HSV
        try:
            image = Image.open(image_path).convert('HSV')
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            raise
        if self.transform:
            image = self.transform(image)
        return image

expansion_ratio = 4
# make a collation function that loads the images, and blanks out the HS channels except for a few random points
# takes a batch of images as a parameter
def collate_function(batch):
    masked_images=[]
    for image in batch:
        for _ in range(expansion_ratio):
            mask = np.zeros_like(image[0])
            num_points = np.randint(0,6)
            total_points = image.size
            random_points = np.random.choice(total_points, size = num_points, replace=False)

            for index in random_points:
                row, col = divmod(index, image.shape[1])
                mask[row, col] = 1

            masked_image = [image[0][mask ==1], image[1][mask ==1], image[2]]
            masked_images.append(torch.tensor(masked_image, dtype=torch.float32))

    return {"images": batch, "masked_images": torch.stack(masked_images)}

# load dataset
def load_dataset(train_folder, test_folder, val_folder, batch_size=batch_size, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalise(mean=[0.5], std=[0.25])
        ])

    # create train, test, and val datasets
    train_dataset = ImageDataset(train_folder, transform=transform)
    test_dataset = ImageDataset(test_folder, transform=transform)
    val_dataset = ImageDataset(val_folder, transform=transform)

    # load datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_function)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_function)

    return train_loader, test_loader, val_loader

# loading dataset
train_loader, test_loader, val_loader = DataLoader(train_folder, test_folder, val_folder, batch_size=batch_size)

# training loop
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet()
model.to(device)

learning_rate = 5e-5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(train_loader):
        # define loss function 
        # define input and 'labels'
        images = batch['images']
        masked_images = batch['masked_images']

        images.to(device)
        masked_images.to(device)

        optimizer.zero_grad()
        
        total_loss= 0.0
        output = model(masked_images)
        for j in range(expansion_ratio):
            loss = criterion(output[j,:,:], images[:,:-1,:,:])
            total_lost += loss
        
        total_loss.backward()
        optimizer.step()