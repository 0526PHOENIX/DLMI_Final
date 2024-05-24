"""
====================================================================================================
Package
====================================================================================================
"""
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.nn import MSELoss, L1Loss
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.classification import Dice

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
====================================================================================================
Pixelwise Loss: L1 Loss
====================================================================================================
"""
def get_pix_loss(predicts, labels):

    return L1Loss().to(DEVICE)(predicts, labels)


"""
====================================================================================================
Gradient Difference Loss
====================================================================================================
"""
def get_gdl_loss(predicts, labels):

    # First Derivative of Predicts
    grad_predicts_x = torch.abs(predicts[:, :, 1:, :] - predicts[:, :, :-1, :])
    grad_predicts_y = torch.abs(predicts[:, :, :, 1:] - predicts[:, :, :, :-1])

    # First Derivative of Labels
    grad_labels_x = torch.abs(labels[:, :, 1:, :] - labels[:, :, :-1, :])
    grad_labels_y = torch.abs(labels[:, :, :, 1:] - labels[:, :, :, :-1])

    # Gradient Difference
    gdl_x = MSELoss().to(DEVICE)(grad_predicts_x, grad_labels_x)
    gdl_y = MSELoss().to(DEVICE)(grad_predicts_y, grad_labels_y)

    return gdl_x + gdl_y


"""
====================================================================================================
Similarity Loss
====================================================================================================
"""
def get_sim_loss(predicts, labels):

    return 1 - StructuralSimilarityIndexMeasure().to(DEVICE)(predicts, labels)


"""
====================================================================================================
MAE: L1 Loss
====================================================================================================
"""
def get_mae(predicts, labels):

    return L1Loss().to(DEVICE)(predicts, labels).item()


"""
====================================================================================================
Head MAE: L1 Loss
====================================================================================================
"""
def get_head(predicts, labels, masks):

    predicts = torch.where(masks, predicts, -1000)

    return L1Loss().to(DEVICE)(predicts, labels).item()


"""
====================================================================================================
Skull MAE: L1 Loss
====================================================================================================
"""
def get_skull(predicts, labels, threshold = 2000):

    predicts = torch.where(predicts > threshold, predicts, -1000)
    labels = torch.where(labels > threshold, labels, -1000)

    return L1Loss().to(DEVICE)(predicts, labels).item()


"""
====================================================================================================
Skull Dice
====================================================================================================
"""
def get_dice(predicts, labels, threshold = 2000):

    predicts = torch.where(predicts > threshold, 1, 0)
    labels = torch.where(labels > threshold, 1, 0)

    return Dice(num_classes = 2, average = 'micro').to(DEVICE)(predicts, labels).item()


"""
====================================================================================================
PSNR
====================================================================================================
"""
def get_psnr(predicts, labels):

    return PeakSignalNoiseRatio().to(DEVICE)(predicts, labels).item()


"""
====================================================================================================
SSIM
====================================================================================================
"""
def get_ssim(predicts, labels):

    return StructuralSimilarityIndexMeasure().to(DEVICE)(predicts, labels).item()


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    image = torch.rand((16, 1, 256, 256))
    label = torch.rand((16, 1, 256, 256))

    ssim = get_sim_loss(image, label)
    print(ssim, ssim.size())