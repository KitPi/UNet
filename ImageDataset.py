from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
from torchvision import transforms
import numpy as np

# dataset
class ImageDataset(Dataset):
    def __init__(self, images_path, transform = None):
        self.image_list = [os.path.join(images_path, im) for im in os.listdir(images_path) if im.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'))]
        self.transform = transform
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, index):
        image_path = self.image_list[index]
        # attempt to open images and convert to LAB
        try:
            image = Image.open(image_path).convert('LAB')
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            raise
        if self.transform:
            image = self.transform(image)
        return image
    


def collate_function(batch):
    hints=[]
    images=[]
    for image in batch:
        #expansion_list = []
        #for _ in range(expansion_ratio):

        mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)

        max_num_points = 150# 125# 100# 75# 50# 25# 15# 10# 5
        num_points = np.random.randint(1, max_num_points) - 1
        total_points = image.shape[0] * image.shape[1]

        random_points = np.random.choice(total_points, size = num_points, replace=False)

        for index in random_points:
            row, col = divmod(index, image.shape[2])
            mask[row, col] = 1

        # Create a masked image
        hint = image.clone()

        # Apply the mask to all three channels of the LAB image
        #for i in range(3):
        hint[0, :, :] = image[0, :, :] * mask #* 100# c,h,w
        hint[1, :, :] = image[1, :, :] * mask #* 127 -128# c,h,w
        hint[2, :, :] = image[2, :, :] * mask #* 127 -128# c,h,w

            
        hints.append(hint)
        images.append(image)

    return {
        "images": torch.stack(images),
        "hints": torch.stack(hints)
    }

# load dataset
#def load_dataset(train_folder, test_folder, val_folder, batch_size=batch_size, transform=None):
def load_dataset(train_folder, batch_size, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0], std=[0.5])
        ])

    # create train, test, and val datasets
    train_dataset = ImageDataset(train_folder, transform=transform)
    #test_dataset = ImageDataset(test_folder, transform=transform)
    #val_dataset = ImageDataset(val_folder, transform=transform)

    # load datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function)

    return train_loader#, test_loader, val_loader
