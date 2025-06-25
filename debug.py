import os 
from PIL import Image
import random

images_path = "dataset/train/"

def main():
    image_list = [os.path.join(images_path, im) for im in os.listdir(images_path) if im.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'))]
    image_path = random image_list
    try:
        image = Image.open(image_path).convert('HSV')
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        raise