"""
====================================================================================================
Package
====================================================================================================
"""
import torch
from torch import Tensor
from torch.nn import MSELoss, L1Loss
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
========================================================================================================
Loss Function & Metrics
========================================================================================================
"""
class Loss():

    """
    ====================================================================================================
    Pixelwise Loss: L1 Loss
    ====================================================================================================
    """
    def get_pix_loss(predicts: Tensor, labels: Tensor) -> Tensor:

        return L1Loss().to(DEVICE)(predicts, labels)

    """
    ====================================================================================================
    Gradient Difference Loss
    ====================================================================================================
    """
    def get_gdl_loss(predicts: Tensor, labels: Tensor) -> Tensor:

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
    def get_sim_loss(predicts: Tensor, labels: Tensor) -> Tensor:

        return 1 - StructuralSimilarityIndexMeasure().to(DEVICE)(predicts, labels)

    """
    ====================================================================================================
    MAE: L1 Loss
    ====================================================================================================
    """
    def get_mae(predicts: Tensor, labels: Tensor) -> float:

        return L1Loss().to(DEVICE)(predicts, labels).item()

    """
    ====================================================================================================
    Head MAE: L1 Loss
    ====================================================================================================
    """
    def get_head(predicts: Tensor, labels: Tensor, masks: Tensor) -> float:

        predicts = torch.where(masks, predicts, -1000)

        return L1Loss().to(DEVICE)(predicts, labels).item()

    """
    ====================================================================================================
    PSNR
    ====================================================================================================
    """
    def get_psnr(predicts: Tensor, labels: Tensor) -> float:

        return PeakSignalNoiseRatio().to(DEVICE)(predicts, labels).item()

    """
    ====================================================================================================
    SSIM
    ====================================================================================================
    """
    def get_ssim(predicts: Tensor, labels: Tensor) -> float:

        return StructuralSimilarityIndexMeasure().to(DEVICE)(predicts, labels).item()


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    image = torch.rand((16, 1, 256, 256))
    label = torch.rand((16, 1, 256, 256))

    pix = Loss.get_pix_loss(image, label)
    print(pix)

    gdl = Loss.get_gdl_loss(image, label)
    print(gdl)

    mae = Loss.get_mae(image, label)
    print(mae)

    psnr = Loss.get_psnr(image, label)
    print(psnr)

    ssim = Loss.get_ssim(image, label)
    print(ssim)