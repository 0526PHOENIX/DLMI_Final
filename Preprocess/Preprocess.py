"""
====================================================================================================
Package
====================================================================================================
"""
import os
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion
from matplotlib import pyplot as plt

import numpy as np
import nibabel as nib


"""
====================================================================================================
Global Constant
====================================================================================================
"""
RAW = ""
DATA = ""


"""
====================================================================================================
Preprocess
====================================================================================================
"""
class Preprocess():

    """
    ================================================================================================
    Critical Parameters
    ================================================================================================
    """
    def __init__(self, filepath = RAW):

        # Check the Path
        if not os.path.exists(DATA):
            os.makedirs(DATA)
        
        # Load Raw Data
        self.images = np.load(os.path.join(filepath, 'MR.npy')).astype('float32')
        self.labels = np.load(os.path.join(filepath, 'CT.npy')).astype('float32')

        # Check File Number
        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError('\n', 'Unequal Amount of images and labels.', '\n')
        
        self.len = self.images.shape[0]

    """
    ================================================================================================
    Main Process: Remove Background
    ================================================================================================
    """
    def main(self):

        buffer_mr = []
        buffer_ct = []
        buffer_tg = []
        for i in range(self.len):

            # Get Data
            image = self.images[i]
            label = self.labels[i]

            # Rotate
            image = np.rot90(image)
            label = np.rot90(label)

            # Thresholding
            binary = (image > 0.0625)

            # Get Connective Component
            component, feature = ndimage.label(binary)

            # Compute Size of Each Component
            size = ndimage.sum(binary, component, range(1, feature + 1))

            # Find Largest Component
            largest_component = np.argmax(size) + 1

            mask = (component == largest_component)

            # Fill Holes in Mask
            mask = binary_dilation(mask, np.ones((15, 15)))
            mask = binary_erosion(mask, np.ones((15, 15)))
            
            # Apply Mask 
            image = np.where(mask, image, 0)
            label = np.where(mask, label, 0)
            mask = np.where(mask, 1, 0)

            # Stack
            buffer_mr.append(image)
            buffer_ct.append(label)
            buffer_tg.append(mask)

        # Get Series from Stack
        result_mr = np.stack(buffer_mr, axis = 0)
        result_ct = np.stack(buffer_ct, axis = 0)
        result_tg = np.stack(buffer_tg, axis = 0)

        # Save MR
        result_mr = nib.Nifti1Image(result_mr, np.eye(4))
        nib.save(result_mr, os.path.join(DATA, 'MR.nii'))

        # Save CT
        result_ct = nib.Nifti1Image(result_ct, np.eye(4))
        nib.save(result_ct, os.path.join(DATA, 'CT.nii'))

        # Save Head Region
        result_tg = nib.Nifti1Image(result_tg, np.eye(4))
        nib.save(result_tg, os.path.join(DATA, 'TG.nii'))

        # Check Progress
        print()
        print('Done')
        print()
        print('===================================================================================')

        return
    
    """
    ================================================================================================
    Check Statistics
    ================================================================================================
    """
    def check(self):

        image = self.images
        label = self.labels

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap = 'gray')
        plt.subplot(1, 2, 2)
        plt.imshow(label, cmap = 'gray')
        plt.show()

        space = "{: <15.2f}\t{: <15.2f}"
        print('MR:', image.shape)
        print(space.format(image.max(), image.min()))
        print(space.format(image.mean(), image.std()))
        print()
        print('CT:', label.shape)
        print(space.format(label.max(), label.min()))
        print(space.format(label.mean(), label.std()))
        print()
        print('===============================================================================')

        return
    

"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':
    
    pre = Preprocess(RAW)
    pre.main()

    pre = Preprocess(DATA)
    pre.check()