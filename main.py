import torch
import torch.nn as nn

def double_convolution(in_channels, out_channels):
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True),
        nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True)
    ) 
    return conv_op

class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()

        # max pool layer
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # down convolution layers
        self.down_convolution_1 = double_convolution(input_channels, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)
        self.down_convolution_4 = double_convolution(256, 512)
        self.down_convolution_5 = double_convolution(512, 1024)

        # up convolution layers
        self.up_convolution_1 = double_convolution(1024, 512)
        self.up_convolution_2 = double_convolution(512, 256)
        self.up_convolution_3 = double_convolution(256, 128)
        self.up_convolution_4 = double_convolution(128, 64)

        # up transpose layers
        self.up_transpose_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_transpose_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_transpose_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_transpose_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        # output layer
        self.out = nn.Conv2d(in_channels=64, out_channels=output_channels, kernel_size=1)

    def forward(self, x):
        
        # down encoding
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)
        down_9 = self.down_convolution_5(down_8)

        # up decoding
        up_1 = self.up_transpose_1(down_9)
        x = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([down_5, up_2], 1))
        up_3 = self.up_transpose_3(x)
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))
        up_4 = self.up_transpose_4(x)
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))

        # output
        out = self.out(x)

        return out

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    input = torch.rand((2, 3, 512, 512))
    #input = torch.rand((1, 3, 256, 256))

    input_channels=3
    output_channels=2

    model = UNet(input_channels=input_channels, output_channels=output_channels)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_params:,} total trainable parameters.")

    output = model(input)
    print(output.shape)

    supplementary = torch.rand(())
    output_layer =  output.detach().numpy()[0].reshape((512,512,2))
    output_layer = np.clip(output_layer, 0, 1)
    output_layer = (output_layer * 255).astype(np.uint8)

    plt.imshow(output_layer[:,:,1], cmap='grey')
    plt.axis('off')
    plt.show()