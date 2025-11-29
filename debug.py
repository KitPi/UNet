import os 
from PIL import Image
import random 
import numpy as np
from torch import nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
from skimage import color

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
#        # attempt to open images and convert to LAB
#        try:
#            image = Image.open(image_path).convert('LAB')
#        except FileNotFoundError:
#            print(f"Image not found: {image_path}")
#            raise
#        if self.transform:
#            image = self.transform(image)
#        return image

from ImageDataset import *

images_path = "dataset/train/"
model_path = "output/"
batch_size = 64
epochs=1
num_example_images =5
img_width =img_height=224

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


    criterion = nn.HuberLoss() #nn.MSELoss()
    #ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    total_images = []
    total_loss =0.0
    for j in range(epochs):
        # load model
        #weights_path = "model.pth"
        model = UNet()
        try:
            weights = torch.load(f'output/checkpoint_{11000}.pth', map_location=device, weights_only=False)
            #weights = torch.load(f'output/model_epoch_1.pth', map_location=device, weights_only=False)
        except FileNotFoundError:
            print(f"Model: output/checkpoint_{11000}.pth does not exist.")
            #print(f"Model: output/model_epoch_1.pth does not exist.")
            raise
        
        model.load_state_dict(weights)#.state_dict())
        model.to(device)
        model.eval()

        device_images = images.to(device)
        device_hints = hints.to(device)
        length = len(device_images)

        output = model(device_images[:, 0, :, :].reshape([length, 1, 224, 224]), device_hints) #batch_size, channels, h, w

        alpha = 0.5
        beta = 0.5
        #loss = alpha * criterion(output, device_images) + beta * (1.0 - ssim(output, device_images)) #+ criterion(output[:,:2,:,:], device_images[:,:2,:,:])
        loss =  criterion(output, device_images[:,1:,:,:])# + beta * (1.0 - ssim(output, device_images[:,1:,:,:]))
        #loss = 1.0 - ssim(output, device_images)

        #loss = criterion(output[:,:2,:,:], device_images[:,:2,:,:]) + 1.0 - ssim(output[:,:2,:,:], device_images[:,:2,:,:])
        #loss = 1.0 - ssim(output, device_images)
        total_loss += loss.item()
        print(f'Total Loss: {total_loss}')

        inferred_images = output.cpu()

        epoch_images = []
        for i, image in enumerate(inferred_images):
            
            #image.shape()
            #image_rgb = Image.fromarray(image.detach().numpy().astype('uint8'), 'LAB')#.convert('RGB')
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
            #image = transforms.functional.to_pil_image(image, mode="LAB")
            output = np.zeros([3,224,224])
            #lab_image = image.detach().numpy()
            
            output[0,:,:] = images[i,0,:,:].detach().numpy()*100.0   # Scale L* channel
            output[1,:,:] = (image[0,:,:].detach().numpy() * 255.0) - 128.0 # Scale a* channel
            output[2,:,:] = (image[1,:,:].detach().numpy() * 255.0) - 128.0  # Scale b* channel

            output = output.transpose(1,2,0)

            #image[:2,:,:] = images[i,:2,:,:]
            rgb_image = color.lab2rgb(output)


            rgb_image_uint8 = (rgb_image*255).astype(np.uint8)
            image_rgb = Image.fromarray(rgb_image_uint8)
            image_rgb.show()



            #output_norm = np.zeros([3,224,224])
#
            #output_norm[0,:,:] = images[i,0,:,:].detach().numpy()*100  # Scale L* channel
            #output_norm[1,:,:] = (image[0,:,:].detach().numpy() * 127) - 128  # Scale a* channel
            #output_norm[2,:,:] = (image[1,:,:].detach().numpy() * 127) - 128  # Scale b* channel
#
            #output_norm = output_norm.transpose(1,2,0)
#
            ##image[:2,:,:] = images[i,:2,:,:]
            #rgb_image_norm = color.lab2rgb(output_norm)
#
#
            #rgb_image_uint8_norm = (rgb_image_norm*255).astype(np.uint8)
            #image_rgb_norm = Image.fromarray(rgb_image_uint8_norm)
            #image_rgb_norm.show()

            #masked_image = transforms.functional.to_pil_image(masked_image, mode="LAB")
            #inferred_image = transforms.functional.to_pil_image(inferred_image, mode="LAB")


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
    #    mim = transforms.functional.to_pil_image(hints[i], mode="LAB")
    #    im = transforms.functional.to_pil_image(images[i], mode="LAB")
    #    out.paste(mim, (0, img_height * i))
    #    out.paste(im, (img_width * (num_example_images+1), img_height * i))
    #out.save(f"{img_out}_combined_img.png")

main()