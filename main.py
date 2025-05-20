import torch
import torch.nn as nn

def double_convolution(in_channels, out_channels):
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True),
        nn.Conv2d(out_channels, out_channels, kernel = 3, paddding = 1),
        nn.ReLU(inplace = True)
    ) 
    return conv_op

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()