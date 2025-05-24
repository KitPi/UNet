from torch.utils.data import Dataset, DataLoader

from main import UNet

model = UNet(input_channels=3, output_channels=2)

# dataset
class CustomDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images=images
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image = self.images[index]
        if self.transform:
            image = self.transform(image)
        return image
    
    