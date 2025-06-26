import os 
from PIL import Image
import random 
import numpy as np

from main import UNet

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import torch

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

images_path = "dataset/test/"
model_path = "output/"
batch_size = 32
epochs=5
num_example_images =5
img_width =img_height=256

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

def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load images
    image_loader = load_dataset(images_path=images_path, batch_size=batch_size)

    batch = next(iter(image_loader))
    
    images = batch["images"][:num_example_images]

    masked_images = batch["masked_images"][:num_example_images]
    #print(masked_images)
    mean = 0.5
    std = 0.25

    #images = images * std + mean
    #masked_images = masked_images * std + mean

    #combined_images = [Image.new("RGB", (img_width * (epochs + 1), img_height)) for _ in range(num_example_images)]

    models = [UNet(input_channels=3, output_channels=2) for _ in range(epochs)]

    total_images = []
    for j in range(epochs):
        # load model
        #weights_path = "model.pth"
        model = UNet(input_channels=3, output_channels=2)
        try:
            weights = torch.load(f'output/model_epoch_{j+1}.pth', map_location=device, weights_only=False)
        except FileNotFoundError:
            print(f"Model: output/model_epoch_{j+1}.pth does not exist.")
            raise
        
        model.load_state_dict(weights)#.state_dict())
        model.to(device)

        device_masked_images = masked_images.to(device)

        output = model(device_masked_images)

        inferred_images = output.cpu()

        epoch_images = []
        for i in range(num_example_images):
            #images = batch['images']
            #masked_images = batch['masked_images']

            #device_images = images.to(device)
            
            #print(inferred_images[0, :, :, :].shape)# = images[0, 2, :, :]
            #print(images[0, 2, :, :].unsqueeze(0).shape)
            inferred_image = torch.cat((inferred_images[i, :, :, :], images[i, 2, :, :].unsqueeze(0)), dim=0)


            #image = images[0]
            #masked_image = masked_images[0]

            # Un-normalise image-tensors
            
            inferred_image = inferred_image * std + mean

            # image-tensor to images
            #image = transforms.functional.to_pil_image(image, mode="HSV")
            #masked_image = transforms.functional.to_pil_image(masked_image, mode="HSV")
            inferred_image = transforms.functional.to_pil_image(inferred_image, mode="HSV")


            #image.convert(mode="RGB").save(f"{img_out}_{i}.png")
            #masked_image.convert(mode="RGB").save(f"{img_out}_masked_{i}.png")
            epoch_images.append(inferred_image.convert(mode="RGB"))#.save(f"{img_out}_inferred_{i}.png")

        total_images.append(epoch_images)
    
    
    img_out = 'examples/'
    out = Image.new("RGB", (img_width * (num_example_images + 2), img_height * num_example_images))
    for i, im_list in enumerate(total_images):
        for j, im in enumerate(im_list):
            out.paste(im, (img_width * (i + 1), img_height *j))
        #out.paste(masked_images[i].convert(mode="RGB"), (img_width * num_example_images, img_height *i))

    images = images * std + mean
    masked_images = masked_images * std + mean

    for i in range(num_example_images):
        mim = transforms.functional.to_pil_image(masked_images[i], mode="HSV")
        im = transforms.functional.to_pil_image(images[i], mode="HSV")
        out.paste(mim, (0, img_height * i))
        out.paste(im, (img_width * (num_example_images+1), img_height * i))
    out.save(f"{img_out}_combined_img.png")

main()