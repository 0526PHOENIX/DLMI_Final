"""
====================================================================================================
Package
====================================================================================================
"""
import os
import random
import numpy as np
import nibabel as nib

import warnings
warnings.filterwarnings('ignore')

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


"""
====================================================================================================
Dataset
====================================================================================================
"""
class Data(Dataset):

    """
    ================================================================================================
    Data Path & Load Data
    ================================================================================================
    """
    def __init__(self, root: str = "", mode: str = "", augment: bool = False) -> None:

        if mode not in ['Train', 'Val', 'Test']:
            raise ValueError('Invalid Mode. Mode Must Be "Train" or "Val" or "Test"')

        # Data Augmentation
        self.augment = augment

        # Filepath
        self.root = root
        self.images_path = os.path.join(self.root, mode, 'MR.nii')
        self.labels_path = os.path.join(self.root, mode, 'CT.nii')
        self.masks_path = os.path.join(self.root, mode, 'TG.nii')

        # Load MR Data
        self.images = nib.load(self.images_path).get_fdata().astype('float32')
        self.images = torch.from_numpy(self.images)

        # Load CT Data
        self.labels = nib.load(self.labels_path).get_fdata().astype('float32')
        self.labels = torch.from_numpy(self.labels)

        # Load TG Data
        self.masks = nib.load(self.masks_path).get_fdata().astype('bool')
        self.masks = torch.from_numpy(self.masks)

        # Check Data Quantity
        if self.images.shape != self.labels.shape:
            raise ValueError('Unequal Amount of Images and Labels.')

    """
    ================================================================================================
    Number of Data
    ================================================================================================
    """
    def __len__(self) -> int:
        
        return self.images.shape[0]

    """
    ================================================================================================
    Get Data
    ================================================================================================
    """
    def __getitem__(self, index: int) -> Tensor:

        # Load MR Data: (1, 256, 256)
        image = self.images[index : index + 1, :, :]
        
        # Load CT Data: (1, 256, 256)
        label = self.labels[index : index + 1, :, :]

        # Load TG Data: (1, 256, 256)
        mask = self.masks[index : index + 1, :, :]

        if self.augment:
            return self.augmentation(image, label, mask)
        else:
            return (image, label, mask)
    
    """
    ================================================================================================
    Data Augmentation
    ================================================================================================
    """
    def augmentation(self, image: Tensor, label: Tensor, mask: Tensor) -> Tensor:

        if random.random() > 0.5:

            image = TF.hflip(image)
            label = TF.hflip(label)
            mask = TF.hflip(mask)
        
        params = transforms.RandomAffine.get_params(degrees = [-3.5, 3.5],
                                                    translate = None,
                                                    scale_ranges = [0.7, 1.3],
                                                    shears = [0.97, 1.03],
                                                    img_size = [192, 192])

        image = TF.affine(image, *params, fill = 0)
        label = TF.affine(label, *params, fill = 0)
        mask = TF.affine(mask, *params, fill = 0)

        return (image, label, mask)
    

"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    filepath = ""

    data = Data(filepath, 'Test')

    for i in range(3):

        index = random.randint(0, len(data) - 1)

        image, label, _ = data[index]
        
        print()
        print('image:', image.shape)
        print(image.min(), image.max())
        print(image.mean(), image.std())
        print('label:', label.shape)
        print(label.min(), label.max())
        print(label.mean(), label.std())
        print()