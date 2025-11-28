from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision import transforms
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from PIL import Image

import os
import torch 
import numpy as np

from main import UNet

writer = SummaryWriter(log_dir='runs')


## parameters
batch_size = 32
train_folder = "dataset/train/"
test_folder = "dataset/test/"
val_folder = "dataset/val/"
output_dir = "output/"


# dataset
#class ImageDataset(Dataset):
#    def __init__(self, images_path, transform = None):
#        self.image_list = [os.path.join(images_path, im) for im in os.listdir(images_path) if im.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'))]
#        self.transform = transform
#    def __len__(self):
#        return len(self.image_list)
#    def __getitem__(self, index):
#        image_path = self.image_list[index]
#        # attempt to open images and convert to LAB
#        try:
#            image = Image.open(image_path).convert('LAB')
#        except FileNotFoundError:
#            print(f"Image not found: {image_path}")
#            raise
#        if self.transform:
#            image = self.transform(image)
#        return image
    
from ImageDataset import ImageDataset, load_dataset


# loading dataset
#train_loader, test_loader, val_loader = load_dataset(train_folder, test_folder, val_folder, batch_size)
train_loader = load_dataset(train_folder, batch_size)

# training loop
num_epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet()#input_channels=3, output_channels=2)

#try:
#    weights = torch.load(f'output/checkpoint_{1500}.pth', map_location=device, weights_only=False)
#except FileNotFoundError:
#    print(f"Model: output/checkpoint_{1500}.pth does not exist.")
#    raise
#model.load_state_dict(weights)#.state_dict())

model.to(device)

learning_rate = 5e-5 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.HuberLoss()#nn.MSELoss()
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(train_loader):
        # define loss function 
        # define input and 'labels'
        images = batch['images']
        length = len(images)
        #print(images[:, 2, :, :].reshape([batch_size, 1, 224, 224]).shape)
        hints = batch['hints']
        #print(hints.shape)

        device_images = images.to(device)
        hints = hints.to(device)
        #model = model.to(device)

        optimizer.zero_grad()
        
        #total_loss= 0.0
        #for j in range(expansion_ratio):
        output = model(device_images[:, 0, :, :].reshape([length, 1, 224, 224]), hints) #batch_size, channels, h, w
        #ssim_loss = 
        alpha = 0.5
        beta = 0.5
        loss = criterion(output, device_images[:,1:,:,:])# + beta * (1.0 - ssim(output, device_images[:,1:,:,:]))
        #loss = criterion(output[:,:2,:,:], images[:,:2,:,:]) + 1.0 - ssim(output[:,:2,:,:], images[:,:2,:,:])
        # batch_size, ?expansion ratio?, channels, h, w :: vs :: batch_size, channels, h, w
        #total_loss += loss

        if i % batch_size ==0:
            print(f'Epoch: [{epoch+1}/{num_epochs}], Batch [{i}], Total loss: {loss.item():.4f}')
        
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)


        if i % 1000 ==0:
            print(f'Checkpoint: [{epoch+1}/{num_epochs}], Batch [{i}]')
            torch.save(model.state_dict(), output_dir + f'checkpoint_{i}.pth')

    ## Save model
    torch.save(model.state_dict(), output_dir + f'model_epoch_{epoch+1}.pth')
    print(f'Model saved at epoch {epoch+1}')
    
writer.close()                              

## Save model
torch.save(model.cpu(), output_dir + 'model.pth')

        