import os 
from PIL import Image
import random 
import numpy as np
from torch import nn
from torchmetrics.image import StructuralSimilarityIndexMeasure

from main import UNet

from ImageDataset import collate_function, load_dataset


from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import torch

# dataset
#class ImageDataset(Dataset):
#    def __init__(self, images_path, transform = None):
#        self.image_list = [os.path.join(images_path, im) for im in os.listdir(images_path) if im.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'))]
#        self.transform = transform
#    def __len__(self):
#        return len(self.image_list)
#    def __getitem__(self, index):
#        image_path = self.image_list[index]
#        # attempt to open images and convert to HSV
#        try:
#            image = Image.open(image_path).convert('HSV')
#        except FileNotFoundError:
#            print(f"Image not found: {image_path}")
#            raise
#        if self.transform:
#            image = self.transform(image)
#        return image

from ImageDataset import *

images_path = "dataset/train/"
model_path = "output/"
batch_size = 32
epochs=1
num_example_images =1
img_width =img_height=224

#def collate_function(batch):
#    masked_images=[]
#    for image in batch:
#        #expansion_list = []
#        #for _ in range(expansion_ratio):
#
#        mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)
#
#        num_points = np.random.randint(1,6)
#        total_points = image.shape[0] * image.shape[1]
#
#        random_points = np.random.choice(total_points, size = num_points, replace=False)
#
#        for index in random_points:
#            row, col = divmod(index, image.shape[2])
#            mask[row, col] = 1
#
#        masked_image = image.clone()
#        masked_image[0] = masked_image[0] * mask
#        masked_image[1] = masked_image[1] * mask
#
#            #expansion_list.append(masked_image)
#            
#        masked_images.append(masked_image)
#
#    return {
#        "images": torch.stack(batch),
#        "masked_images": torch.stack(masked_images)
#    }
#
#
#def load_dataset(images_path, batch_size=batch_size, transform=None):
#    if transform is None:
#        transform = transforms.Sequential([
#            transforms.Resize((256, 256)),
#            transforms.ToTensor(),
#            transforms.Normalize(mean=[0.5], std=[0.25])
#        ])
#
#    test_dataset = ImageDataset(images_path, transform=transform)
#    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function)
#    return test_loader

def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load images
    image_loader = load_dataset(images_path, batch_size)

    batch = next(iter(image_loader))
    
    images = batch["images"][:num_example_images]

    #masked_images = batch["masked_images"][:num_example_images]
    hints = batch["hints"][:num_example_images]
    #print(masked_images)
    mean = 0.5
    std = 0.25

    #images = images * std + mean
    #masked_images = masked_images * std + mean

    #combined_images = [Image.new("RGB", (img_width * (epochs + 1), img_height)) for _ in range(num_example_images)]


    criterion = nn.MSELoss()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    total_images = []
    total_loss =0.0
    for j in range(epochs):
        # load model
        #weights_path = "model.pth"
        model = UNet()
        try:
            weights = torch.load(f'output/checkpoint_{12000}.pth', map_location=device, weights_only=False)
            #weights = torch.load(f'output/model_epoch_1.pth', map_location=device, weights_only=False)
        except FileNotFoundError:
            print(f"Model: output/checkpoint_{12000}.pth does not exist.")
            #print(f"Model: output/model_epoch_1.pth does not exist.")
            raise
        
        model.load_state_dict(weights)#.state_dict())
        model.to(device)
        model.eval()

        device_images = images.to(device)
        device_hints = hints.to(device)
        length = len(device_images)

        output = model(device_images[:, 2, :, :].reshape([length, 1, 224, 224]), device_hints) #batch_size, channels, h, w

        alpha = 0.5
        beta = 0.5
        loss = alpha * criterion(output, device_images) + beta * (1.0 - ssim(output, device_images)) #+ criterion(output[:,:2,:,:], device_images[:,:2,:,:])
        #loss = 1.0 - ssim(output, device_images)

        #loss = criterion(output[:,:2,:,:], device_images[:,:2,:,:]) + 1.0 - ssim(output[:,:2,:,:], device_images[:,:2,:,:])
        #loss = 1.0 - ssim(output, device_images)
        total_loss += loss.item()
        print(f'Total Loss: {total_loss}')

        inferred_images = output.cpu()

        epoch_images = []
        for i, image in enumerate(inferred_images):
            
            #image.shape()
            #image_rgb = Image.fromarray(image.detach().numpy().astype('uint8'), 'HSV')#.convert('RGB')
            #image_rgb.show()
            #images = batch['images']
            #masked_images = batch['masked_images']

            #device_images = images.to(device)
            
            #print(inferred_images[0, :, :, :].shape)# = images[0, 2, :, :]
            #print(images[0, 2, :, :].unsqueeze(0).shape)
            #inferred_image = inferred_images[i, :, :, :]#torch.cat((inferred_images[i, :, :, :], images[i, 2, :, :]), dim=1)


            #image = images[0]
            #masked_image = masked_images[0]

            # Un-normalise image-tensors
            
            #inferred_image = inferred_image * std + mean

            # image-tensor to images
            image = transforms.functional.to_pil_image(image, mode="HSV")
            #image[:2,:,:] = images[i,:2,:,:]
            image.show()

            #masked_image = transforms.functional.to_pil_image(masked_image, mode="HSV")
            #inferred_image = transforms.functional.to_pil_image(inferred_image, mode="HSV")


            #image.convert(mode="RGB").save(f"{img_out}_{i}.png")
            #masked_image.convert(mode="RGB").save(f"{img_out}_masked_{i}.png")
            #epoch_images.append(inferred_image.convert(mode="RGB"))#.save(f"{img_out}_inferred_{i}.png")

        #total_images.append(epoch_images)
    
    
    #img_out = 'examples/'
    #out = Image.new("RGB", (img_width * (num_example_images + 2), img_height * num_example_images))
    #for i, im_list in enumerate(total_images):
    #    for j, im in enumerate(im_list):
    #        out.paste(im, (img_width * (i + 1), img_height *j))
    #    #out.paste(masked_images[i].convert(mode="RGB"), (img_width * num_example_images, img_height *i))
#
    ##images = images * std + mean
    ##masked_images = masked_images * std + mean
#
    #for i in range(num_example_images):
    #    mim = transforms.functional.to_pil_image(hints[i], mode="HSV")
    #    im = transforms.functional.to_pil_image(images[i], mode="HSV")
    #    out.paste(mim, (0, img_height * i))
    #    out.paste(im, (img_width * (num_example_images+1), img_height * i))
    #out.save(f"{img_out}_combined_img.png")

main()