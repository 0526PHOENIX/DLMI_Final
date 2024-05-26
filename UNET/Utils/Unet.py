"""
====================================================================================================
Package
====================================================================================================
"""
import torch
import torch.nn as nn
from torchvision.transforms import Resize
from torchsummary import summary


"""
====================================================================================================
Residual Block 2D
====================================================================================================
"""
class Res2d(nn.Module):

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
Initialization Block 2D
====================================================================================================
"""
class Init2d(nn.Module):

    def __init__(self, filters):

        super().__init__()

        # Convolution -> Dropout -> Residual Block
        self.conv = nn.Conv2d(1, filters, kernel_size = 3, padding = 1, bias = False)
        self.drop = nn.Dropout2d(0.2)
        self.res = Res2d(filters)

    def forward(self, img_in):

        img_out = self.conv(img_in)
        img_out = self.drop(img_out)
        img_out = self.res(img_out)

        return img_out


"""
====================================================================================================
Downsampling Block 2D
====================================================================================================
"""
class Down2d(nn.Module):

    def __init__(self, filters):

        super().__init__()

        # Downsampling -> Residual Block
        self.down = nn.Conv2d(filters // 2, filters, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.res = Res2d(filters)

    def forward(self, img_in_1, img_in_2):

        img_out = self.down(img_in_1)

        # Jump Connection
        img_out += Resize((img_out.shape[2], img_out.shape[3]))(img_in_2)
        img_out = self.res(img_out)

        return img_out


"""
====================================================================================================
Middle Block 2D
====================================================================================================
"""
class Mid2d(nn.Module):

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
Upsampling Block 2D
====================================================================================================
"""
class Up2d(nn.Module):

    def __init__(self, filters):

        super().__init__()

        # Convolution -> Upsampling -> Residual Block
        self.conv = nn.Conv2d(filters * 2, filters, kernel_size = 1, padding = 0, bias = False)
        self.up = nn.ConvTranspose2d(filters, filters, kernel_size = 2, stride = 2, bias = False)
        self.res = Res2d(filters)
    
    def forward(self, img_in_1, img_in_2, img_in_3):

        img_out = self.conv(img_in_1)
        img_out = self.up(img_out)

        # Jump Connection
        img_out += Resize((img_out.shape[2], img_out.shape[3]))(img_in_2)
        img_out += img_in_3
        img_out = self.res(img_out)

        return img_out


"""
====================================================================================================
Final Block 2D
====================================================================================================
"""
class Final2d(nn.Module):

    def __init__(self, filters):

        super().__init__()

        self.final_block = nn.Sequential(nn.Conv2d(filters, 1, kernel_size = 1, bias = False),
                                         nn.Tanh())
    
    def forward(self, img_in):

        img_out = self.final_block(img_in)

        return img_out


"""
====================================================================================================
Unet 2D
====================================================================================================
"""
class Unet2d(nn.Module):

    def __init__(self, depth = 5, bottle = 9):

        super().__init__()

        self.depth = depth
        self.bottle = bottle

        # Number of Filters
        self.filters = [pow(2, i + 4) for i in range(depth)]

        # Initialization
        self.init = Init2d(self.filters[0])

        # Downsampling
        self.down = nn.Sequential(*[Down2d(filters) for filters in self.filters[1 : ]])

        # Bottleneck
        self.mid = nn.Sequential(*[Mid2d(self.filters[-1]) for _ in range(self.bottle)])
        
        # Upsampling
        self.up = nn.Sequential(*[Up2d(filters) for filters in self.filters[-2 :  : -1]])

        # Ouput
        self.final = Final2d(self.filters[0])
    
    def forward(self, img_in):
        
        # Initialization
        init = self.init(img_in)

        self.len = self.depth - 1

        # Downsampling
        down = []
        for i in range(self.len):

            if i == 0:

                down.append(self.down[i](init, img_in))

            else:

                down.append(self.down[i](down[i - 1], img_in))

        # Bottleneck

        mid = []
        for i in range(self.bottle):

            if i == 0:

                mid.append(self.mid[i](down[-1]))
            
            else:

                mid.append(self.mid[i](mid[i - 1]))

        # Upsampling
        up = []
        for i in range(self.len):

            if i == 0:

                up.append(self.up[i](mid[-1], img_in, down[self.len - i - 2]))

            elif i == self.len - 1:

                up.append(self.up[i](up[i - 1], img_in, init))

            else:

                up.append(self.up[i](up[i - 1], img_in, down[self.len - i - 2]))

        # Ouput
        img_out = self.final(up[-1])

        return img_out


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print('\n' + 'Training on: ' + str(device) + '\n')
    
    model_1 = Unet2d(depth = 5, bottle = 9).to(device = device)
    print(summary(model_1, input_size = (1, 256, 256), batch_size = 6))