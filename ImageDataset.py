from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
from torchvision import transforms

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


def load_dataset(images_path, batch_size=batch_size, transform=None):
    if transform is None:
        transform = transforms.Sequential([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.25])
        ])

    test_dataset = ImageDataset(images_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function)
    return test_loader