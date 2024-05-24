"""
====================================================================================================
Package
====================================================================================================
"""
import torch
import torch.nn as nn
from torchsummary import summary


"""
====================================================================================================
Residual Block
====================================================================================================
"""
class Res(nn.Module):

    def __init__(self, filters):

        super().__init__()

        # (Normalization -> Activation -> Convolution) * 2
        self.res_block = nn.Sequential(nn.BatchNorm2d(filters), 
                                       nn.LeakyReLU(0.01),
                                       nn.Conv2d(filters, filters, kernel_size = 3, padding = 1, bias = False),
                                       nn.BatchNorm2d(filters),
                                       nn.LeakyReLU(0.01),
                                       nn.Conv2d(filters, filters, kernel_size = 3, padding = 1, bias = False))

    def forward(self, img_in):

        img_out = self.res_block(img_in)

        # Jump Connection
        return img_in + img_out


"""
====================================================================================================
Initialization Block
====================================================================================================
"""
class Init(nn.Module):

    def __init__(self, filters):

        super().__init__()

        # Convolution -> Dropout -> Residual Block
        self.conv = nn.Conv2d(1, filters, kernel_size = 3, padding = 1, bias = False)
        self.drop = nn.Dropout2d(0.2)
        self.res = Res(filters)

    def forward(self, img_in):

        img_out = self.conv(img_in)
        img_out = self.drop(img_out)
        img_out = self.res(img_out)

        return img_out


"""
====================================================================================================
Downsampling Block
====================================================================================================
"""
class Down(nn.Module):

    def __init__(self, filters):

        super().__init__()

        # Downsampling -> Residual Block
        self.down = nn.Conv2d(filters // 2, filters, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.res = Res(filters)

    def forward(self, img_in):

        img_out = self.down(img_in)
        img_out = self.res(img_out)

        return img_out


"""
====================================================================================================
Middle Block
====================================================================================================
"""
class Mid(nn.Module):

    def __init__(self, filters):

        super().__init__()

        # Normalization -> Activation -> Convolution -> Dropout -> Normalization -> Convolution
        self.bottle_block = nn.Sequential(nn.BatchNorm2d(filters),
                                          nn.LeakyReLU(0.01),
                                          nn.Conv2d(filters, filters, kernel_size = 3, padding = 1, bias = False),
                                          nn.Dropout2d(0.5),
                                          nn.BatchNorm2d(filters),
                                          nn.Conv2d(filters, filters, kernel_size = 3, padding = 1, bias = False))
    
    def forward(self, img_in):

        img_out = self.bottle_block(img_in)

        # Jump Connection
        return img_in + img_out


"""
====================================================================================================
Upsampling Block
====================================================================================================
"""
class Up(nn.Module):

    def __init__(self, filters):

        super().__init__()

        # Convolution -> Upsampling -> Residual Block
        self.conv = nn.Conv2d(filters * 2, filters, kernel_size = 1, padding = 0, bias = False)
        self.up = nn.ConvTranspose2d(filters, filters, kernel_size = 2, stride = 2, bias = False)
        self.res = Res(filters)
    
    def forward(self, img_in_1, img_in_2):

        img_out = self.conv(img_in_1)
        img_out = self.up(img_out)

        # Jump Connection
        img_out += img_in_2
        img_out = self.res(img_out)

        return img_out


"""
====================================================================================================
Final Block
====================================================================================================
"""
class Final(nn.Module):

    def __init__(self, filters):

        super().__init__()

        self.final_block = nn.Sequential(nn.Conv2d(filters, 1, kernel_size = 1, bias = False),
                                         nn.Tanh())
    
    def forward(self, img_in):

        img_out = self.final_block(img_in)

        return img_out


"""
====================================================================================================
Unet
====================================================================================================
"""
class Unet(nn.Module):

    def __init__(self):

        super().__init__()

        # Number of Filters
        self.filters = [16, 32, 64, 128, 256]

        # Initialization
        self.init = Init(self.filters[0])

        # Downsampling
        self.down_1 = Down(self.filters[1])
        self.down_2 = Down(self.filters[2])
        self.down_3 = Down(self.filters[3])
        self.down_4 = Down(self.filters[4])

        # Bottleneck
        self.mid_1 = Mid(self.filters[4])
        self.mid_2 = Mid(self.filters[4])
        self.mid_3 = Mid(self.filters[4])
        self.mid_4 = Mid(self.filters[4])
        self.mid_5 = Mid(self.filters[4])
        self.mid_6 = Mid(self.filters[4])
        self.mid_7 = Mid(self.filters[4])
        self.mid_8 = Mid(self.filters[4])
        self.mid_9 = Mid(self.filters[4])
        
        # Upsampling
        self.up_4 = Up(self.filters[3])
        self.up_3 = Up(self.filters[2])
        self.up_2 = Up(self.filters[1])
        self.up_1 = Up(self.filters[0])

        # Ouput
        self.final = Final(self.filters[0])
    
    def forward(self, img_in):
        
        # Initialization
        init = self.init(img_in)

        # Downsampling
        down_1 = self.down_1(init)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)

        # Bottleneck
        mid = self.mid_1(down_4)
        mid = self.mid_2(mid)
        mid = self.mid_3(mid)
        mid = self.mid_4(mid)
        mid = self.mid_5(mid)
        mid = self.mid_6(mid)
        mid = self.mid_7(mid)
        mid = self.mid_8(mid)
        mid = self.mid_9(mid)

        # Upsampling
        up_4 = self.up_4(mid, down_3)
        up_3 = self.up_3(up_4, down_2)
        up_2 = self.up_2(up_3, down_1)
        up_1 = self.up_1(up_2, init)

        # Ouput
        img_out = self.final(up_1)

        return img_out


"""
====================================================================================================
Pretrained Unet
====================================================================================================
"""
def Pretrain():

    model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained = True, scale = 0.5, verbose = False)
    
    in_filters = model.inc.double_conv[3].out_channels
    model.inc = Init(in_filters)

    out_filters = model.outc.conv.in_channels
    model.outc = Final(out_filters)

    return model


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print('Training on: ' + str(device) + '\n')
    
    # model_1 = Unet().to(device = device)
    # print(summary(model_1, input_size = (1, 256, 256), batch_size = 2))

    model_2 = Pretrain().to(device = device)
    print(summary(model_2, input_size = (1, 192, 192), batch_size = 2))