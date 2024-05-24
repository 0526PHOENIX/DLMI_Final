"""
====================================================================================================
Package
====================================================================================================
"""
import os
import random
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset


"""
====================================================================================================
Dataset
====================================================================================================
"""
class Data(Dataset):

    def __init__(self, root = "", mode = ""):

        # Filepath
        self.root = root
        self.images_path = os.path.join(self.root, mode, 'MR.npy')
        self.labels_path = os.path.join(self.root, mode, 'CT.npy')
        self.masks_path = os.path.join(self.root, mode, 'TG.npy')

        # Load MR Data: (570, 256, 256)
        self.images = np.load(self.images_path).astype('float32')
        self.images = torch.from_numpy(self.images)

        # Load CT Data: (570, 256, 256)
        self.labels = np.load(self.labels_path).astype('float32')
        self.labels = torch.from_numpy(self.labels)

        # Load TG Data: (570, 256, 256)
        self.masks = np.load(self.masks_path).astype('bool')
        self.masks = torch.from_numpy(self.masks)

        # Check Data Quantity
        if self.images.shape != self.labels.shape:
            raise ValueError('Unequal Amount of Images and Labels.')

    def __len__(self):
        
        return self.images.shape[0]

    def __getitem__(self, index):

        # Load MR Data: (1, 256, 256)
        image = self.images[index : index + 1, :, :]
        
        # Load CT Data: (1, 256, 256)
        label = self.labels[index : index + 1, :, :]

        # Load TG Data: (1, 256, 256)
        mask = self.masks[index : index + 1, :, :]

        return (image, label, mask)
    

"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    filepath = ""

    train = Data(filepath, 'Train')
    val = Data(filepath, 'Val')
    test = Data(filepath, 'Test')

    for i in range(5):

        index = random.randint(0, len(train) - 1)

        image, label, mask = train[index]
        
        print()
        print('image:')
        print(image.min(), image.max())
        print(image.mean(), image.std())
        print('label:')
        print(label.min(), label.max())
        print(label.mean(), label.std())
        print()