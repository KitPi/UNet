
import numpy as np
from PIL import Image
import glob

batch_size = 32

train_folder = "dataset/io/*"


images = [Image.open(file).convert("HSV") for file in glob.glob(train_folder)]


image = np.asarray(images[0])

print(image.shape)

mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
num_points = 150000 #np.random.randint(1,6)
total_points = image.shape[0] * image.shape[1]
random_points = np.random.choice(total_points, size = num_points, replace=False)

for index in random_points:
    row, col = divmod(index, image.shape[1])
    mask[row, col] = 1


# Create a masked image
hints = image.copy()

# Apply the mask to all three channels of the HSV image
for i in range(3):
    hints[:, :, i] = image[:, :, i] * mask # h,w,c

# Convert back to RGB for display
hue_hints = Image.fromarray(hints[:, :, 0].astype('uint8'))#.convert('RGB')
sat_hints = Image.fromarray(hints[:, :, 1].astype('uint8'))#.convert('RGB')
val = Image.fromarray(hints[:, :, 2].astype('uint8'))
image_rgb = Image.fromarray(image.astype('uint8'), 'HSV')#.convert('RGB')

# Show the masked image
#masked_image_rgb.show()
hue_hints.show()
sat_hints.show()
val.show()
image_rgb.show()