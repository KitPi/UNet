from torch.utils.data import Dataset, DataLoader
from skimage import transforms
import os
from PIL import Image

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

# make a collation function that loads the images in HSV colour space, and blanks out the HS channels except for a few random points
# takes a batch of images as a parameter
def collate_function(batch):
    return image

# load dataset
def load_dataset(dataset_folder, train_list, test_list, val_list, batch_size=batch_size, transform=None):
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