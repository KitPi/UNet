import torch
import torch.nn as nn

# H_out = (H_in + 2 * padding - kernel_size) / (stride) + 1

def down_convolution(in_channels, out_channels):
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True),
        nn.Dropout(0.25),
        nn.Conv2d(out_channels, out_channels, kernel_size = 5, padding = 2, stride =2),
        nn.ReLU(inplace = True),
        nn.BatchNorm2d(out_channels)
    ) 
    return conv_op

def hints_down_convolution(in_channels, out_channels):
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 5, padding = 2, stride =2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True)
    ) 
    return conv_op

def up_convolution(in_channels, out_channels):
    # H_out = (H_in + 2 * padding - kernel_size) / (stride) + 1
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True),
        nn.BatchNorm2d(out_channels),
    ) 
    return conv_op

def self_conv(in_channels, out_channels):
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True),
        nn.BatchNorm2d(out_channels),
        #nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
        #nn.ReLU(inplace = True),
        #nn.BatchNorm2d(out_channels),
    )
    return conv_op

def out_layer(in_channels, out_channels):
    conv_op = nn.Sequential(
        #nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
        #nn.BatchNorm2d(out_channels),
        #nn.Sigmoid()
        #
        nn.Conv2d(in_channels, 16, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 2, kernel_size = 3, padding = 1),
        #nn.BatchNorm2d(2),
        nn.Sigmoid(),
    )
    return conv_op

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # max number of hint points
        #self.max_num_points = max_num_points

        # max pool layer
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(0.2)

        # down convolution layers
        self.down_convolution_1 = down_convolution(4, 64) # h/2, w/2, 64
        self.down_convolution_2 = down_convolution(80, 128) # h/4, w/4, 128
        self.down_convolution_3 = down_convolution(160, 256) # h/8, w/8, 256
        self.down_convolution_4 = down_convolution(320, 512) # h/16, w/16, 512
        self.down_convolution_5 = down_convolution(640, 1024) # h/16, w/16, 1024

        # hints down convolution layers
        self.hint_down_conv1 = hints_down_convolution(3, 16) # h/2, w/2, 16
        self.hint_down_conv2 = hints_down_convolution(16, 32) # h/4, w/4, 32
        self.hint_down_conv3 = hints_down_convolution(32, 64) # h/8, w/8, 64
        self.hint_down_conv4 = hints_down_convolution(64, 128) # h/16, w/16, 128

        # up convolution layers
        self.up_convolution_1 = up_convolution(1024, 512)
        self.up_convolution_2 = up_convolution(512, 256)
        self.up_convolution_3 = up_convolution(256, 128)
        self.up_convolution_4 = up_convolution(128, 64)

        # up transpose layers
        # H_out = (H_in + 2 * padding - kernel_size) / (stride) + 1
        self.up_transpose_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_transpose_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_transpose_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_transpose_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_transpose_5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)

        # hint self convolution layers
        self.hint_self_conv = self_conv(3, 3)

        # image self convolution layers
        self.image_self_conv1 = self_conv(64, 64)
        self.image_self_conv2 = self_conv(128, 128)
        self.image_self_conv3 = self_conv(256, 256)
        self.image_self_conv4 = self_conv(512, 512)
        

        # output layer
        self.out = out_layer(35, 2)

    def forward(self, images, hints):

        # self convolve hints
        hints_2 = self.hint_self_conv(hints) # h, w, 3
        #hints_2 = self.max_pool2d(hints_1) # h/2, w/2, 3

        hints_3 = self.hint_down_conv1(hints_2) # h/2, w/2, 16
        #hints_4 = self.max_pool2d(hints_3) # h/2, w/2, 16
#
        hints_4 = self.hint_down_conv2(hints_3) # h/4, w/4, 32
        #hints_6 = self.max_pool2d(hints_5) # h/4, w/4, 32
#
        hints_5 = self.hint_down_conv3(hints_4) # h/8, w/8, 64
        #hints_8 = self.max_pool2d(hints_7) # h/8, w/8, 64
#
        hints_6 = self.hint_down_conv4(hints_5) # h/16, w/16, 128
        #hints_10 = self.max_pool2d(hints_9) # h/16, w/16, 128

        #hints_7 = self.hint_down_conv4(hints_6) # h/32, w/32, 256

        # concat hints and images
        in_1 = torch.cat([hints_2, images], 1) # h, w, 4 = ( 3 + 1 )
        
        # down encoding
        down_1 = self.down_convolution_1(in_1) # h/2, w/2, 64
        across1 = self.image_self_conv1(down_1) # h/2, w/2, 64 
        #down_2 = self.max_pool2d(down_1) # h/2, w/2, 64
        #down_2 = self.dropout(down_2)

        in_2 = torch.cat([hints_3, down_1], 1) # h/2, w/2, 80 = ( 16 + 64 )
        
        down_3 = self.down_convolution_2(in_2) # h/4, w/4, 128
        across2 = self.image_self_conv2(down_3) # h/4, w/4, 128
        #down_4 = self.max_pool2d(down_3) # h/4, w/4, 128
        #down_4 = self.dropout(down_4)

        in_3 = torch.cat([hints_4, down_3], 1) # h/4, w/4, 160 = ( 32 + 128 )

        down_5 = self.down_convolution_3(in_3) # h/8, w/8, 256
        across3 = self.image_self_conv3(down_5) # h/8, w/8, 256 
        #down_6 = self.max_pool2d(down_5) # h/8, w/8, 256
        #down_6 = self.dropout(down_6)

        in_4 = torch.cat([hints_5, down_5], 1) # h/8, w/8, 320 = ( 64 + 256 )

        down_7 = self.down_convolution_4(in_4) # h/16, w/16, 512
        across4 = self.image_self_conv4(down_7) # h/16, w/16, 512 
        #down_8 = self.max_pool2d(down_7) # h/16, w/16, 512
        #down_8 = self.dropout(down_8)

        in_5 = torch.cat([hints_6, down_7], 1) # h/16, w/16, 640 = ( 128 + 512 )

        down_9 = self.down_convolution_5(in_5) # h/32, w/32, 1024

        # up decoding
        up_1 = self.up_transpose_1(down_9) # h/16, w/16, 512
        cons1 = torch.cat([across4, up_1], 1) # h/16, w/16, 1024 = ( 512 + 512 )
        x = self.up_convolution_1(cons1) # h/16, w/16, 512

        up_2 = self.up_transpose_2(x) # h/8, w/8, 256
        cons2 = torch.cat([across3, up_2], 1) # h/8, w/8, 512 = ( 256 + 256 )
        x = self.up_convolution_2(cons2) # h/8, w/8, 256

        up_3 = self.up_transpose_3(x) # h/4, w/4, 128
        cons3 = torch.cat([across2, up_3], 1) # h/4, w/4, 256 = ( 128 + 128 )
        x = self.up_convolution_3(cons3) # h/4, w/4, 128

        up_4 = self.up_transpose_4(x) # h/2, w/2, 64
        cons4 = torch.cat([across1, up_4], 1) # h/2, w/2, 128 = ( 64 + 64 )
        x = self.up_convolution_4(cons4) # h/2, w/2, 64

        up_5 = self.up_transpose_5(x) # h, w, 32
        cons5 = torch.cat([hints_2, up_5],1) # h, w, 35

        # output
        out = self.out(cons5)

        return out

if __name__ == '__main__':
    # unit tests
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage import color

    images = torch.rand((5, 1, 224, 224))
    hints = torch.rand((5, 3, 224, 224)) # b c h w
    #input = torch.rand((1, 3, 256, 256))

    input_channels=3
    output_channels=2

    model = UNet()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_params:,} total trainable parameters.")

    output = model(images, hints) #batch_size, channels, h, w
    print(output.shape)

    #supplementary = torch.rand(())
    #output_layer =  output.detach().numpy()[0].reshape((224,224,2))
#
    #output = np.zeros([3,224,224])
    ##lab_image = image.detach().numpy()
    #
    #output[0,:,:] = images[i,0,:,:].detach().numpy()*50  # Scale L* channel
    #output[1,:,:] = (image[0,:,:].detach().numpy() * 63) - 64  # Scale a* channel
    #output[2,:,:] = (image[1,:,:].detach().numpy() * 63) - 64  # Scale b* channel
    #output = output.transpose(1,2,0)
    ##image[:2,:,:] = images[i,:2,:,:]
    #rgb_image = color.lab2rgb(output)
    #rgb_image_uint8 = (rgb_image*255).astype(np.uint8)
    #image_rgb = Image.fromarray(rgb_image_uint8)
    #image_rgb.show()
    #output_layer = np.clip(output_layer, 0, 1)
    #output_layer = (output_layer * 255).astype(np.uint8)
##
    #plt.imshow(output_layer, cmap='LAB')
    #plt.axis('off')
    #plt.show()