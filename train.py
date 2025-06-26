from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from PIL import Image

import os
import torch 
import numpy as np

from main import UNet

writer = SummaryWriter(log_dir='runs')

# load models
model = UNet(input_channels=3, output_channels=2)

## parameters
batch_size = 32
train_folder = "dataset/train/"
test_folder = "dataset/test/"
val_folder = "dataset/val/"
output_dir = "output/"


# dataset
class ImageDataset(Dataset):
    def __init__(self, images_path, transform = None):
        self.image_list = [os.path.join(images_path, im) for im in os.listdir(images_path) if im.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'))]
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
        #expansion_list = []
        #for _ in range(expansion_ratio):

        mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)

        num_points = np.random.randint(1,6)
        total_points = image.shape[0] * image.shape[1]

        random_points = np.random.choice(total_points, size = num_points, replace=False)

        for index in random_points:
            row, col = divmod(index, image.shape[2])
            mask[row, col] = 1

        masked_image = image.clone()
        masked_image[0] = masked_image[0] * mask
        masked_image[1] = masked_image[1] * mask

            #expansion_list.append(masked_image)
            
        masked_images.append(masked_image)

    return {
        "images": torch.stack(batch),
        "masked_images": torch.stack(masked_images)
    }

# load dataset
def load_dataset(train_folder, test_folder, val_folder, batch_size=batch_size, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.25])
        ])

    # create train, test, and val datasets
    train_dataset = ImageDataset(train_folder, transform=transform)
    test_dataset = ImageDataset(test_folder, transform=transform)
    val_dataset = ImageDataset(val_folder, transform=transform)

    # load datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function)

    return train_loader, test_loader, val_loader

# loading dataset
train_loader, test_loader, val_loader = load_dataset(train_folder, test_folder, val_folder, batch_size)

# training loop
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(input_channels=3, output_channels=2)
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
        #print(images.shape)
        masked_images = batch['masked_images']
        #print(masked_images.shape)

        images = images.to(device)
        masked_images = masked_images.to(device)
        #model = model.to(device)

        optimizer.zero_grad()
        
        total_loss= 0.0
        #for j in range(expansion_ratio):
        output = model(masked_images[:,:,:,:])
        loss = criterion(output, images[:,:2,:,:])
        # batch_size, ?expansion ratio?, channels, h, w :: vs :: batch_size, channels, h, w
        #total_loss += loss
        
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

        if i % batch_size ==0:
            print(f'Epoch: [{epoch+1}/{num_epochs}], Batch [{i}], Total loss: {loss.item():.4f}')

    ## Save model
    torch.save(model.state_dict(), output_dir + f'model_epoch_{epoch+1}.pth')
    print(f'Model saved at epoch {epoch+1}')
    
writer.close()                              

## Save model
torch.save(model.cpu(), output_dir + 'model.pth')

        