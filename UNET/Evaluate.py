"""
====================================================================================================
Package
====================================================================================================
"""
import os
import random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import warnings
warnings.filterwarnings('ignore')

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Utils import *


"""
====================================================================================================
Global Constant
====================================================================================================
"""
METRICS = 4
METRICS_MAE = 0
METRICS_HEAD = 1
METRICS_PSNR = 2
METRICS_SSIM = 3


"""
====================================================================================================
Evaluate
====================================================================================================
"""
class Evaluate():
    
    """
    ================================================================================================
    Initialize Critical Parameters
    ================================================================================================
    """
    def __init__(self,
                 depth: int = 5,
                 bottle: int = 9,
                 data: str = "C:/Users/user/Desktop/DLMI/Data",
                 eva: str = "C:/Users/user/Desktop/DLMI/UNET/Evaluate",
                 weight: str = "",
                 *args,
                 **kwargs) -> None:

        # Evaluating Device: CPU(cpu) or GPU(cuda)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        print('\n' + 'Evaluating on: ' + str(self.device))

        # Model
        self.depth = depth
        self.bottle = bottle

        # File Path
        self.data = data
        self.eva = eva
        self.weight = weight

        return

    """
    ================================================================================================
    TensorBorad
    ================================================================================================
    """  
    def init_tensorboard(self) -> None:

        # Metrics Filepath
        log_dir = os.path.join(self.eva, self.time)

        # Tensorboard Writer
        self.val_writer = SummaryWriter(log_dir + '/Val')
        self.test_writer = SummaryWriter(log_dir + '/Test')

        return
    
    """
    ================================================================================================
    Initialize Testing Data Loader
    ================================================================================================
    """
    def init_dl(self) -> None:

        # Validation
        val_ds = Data(root = self.data, mode = 'Val')
        self.val_dl = DataLoader(val_ds, batch_size = 64, shuffle = True, drop_last = False,
                                 num_workers = 4, pin_memory = True)

        # Testing
        test_ds = Data(root = self.data, mode = 'Test')
        self.test_dl = DataLoader(test_ds, batch_size = 64, shuffle = True, drop_last = False,
                                  num_workers = 4, pin_memory = True)
        
        return 

    """
    ================================================================================================
    Load Model Parameter and Hyperparameter
    ================================================================================================
    """
    def load_model(self) -> None:

        if os.path.isfile(self.weight):

            # Get Checkpoint Information
            checkpoint = torch.load(self.weight)
            print('\n' + 'Model Trained at: ' + checkpoint['time'])
            print('\n' + 'Model Saved at Epoch: ' + str(checkpoint['epoch']))

            # Get Time Stamp
            self.time = checkpoint['time']

            # Model: Unet
            self.model = Unet(depth = self.depth, bottle = self.bottle).to(self.device)

            self.model.load_state_dict(checkpoint['model_state'])

            # Tensorboard
            self.init_tensorboard()

        else:

            raise ValueError('Illegal Weight Path')

        return

    """
    ================================================================================================
    Main Evaluating Function
    ================================================================================================
    """
    def main(self) -> None:

        # Data Loader
        self.init_dl()

        # Get Checkpoint
        self.load_model()

        # Validate Model
        print('\n' + 'Validation: ')
        metrics_val = self.evaluation('val')
        self.print_result(metrics_val)
        self.save_images('val')

        # Evaluate Model
        print('\n' + 'Testing: ')
        metrics_test = self.evaluation('test')
        self.print_result(metrics_test)
        self.save_images('test')

        return

    """
    ================================================================================================
    Evaluation Loop
    ================================================================================================
    """
    def evaluation(self, mode: str) -> Tensor:

        with torch.no_grad():

            # Model: Validation State
            self.model.eval()

            # Get Data Loader
            dl = getattr(self, mode + '_dl')

            # Buffer for Matrics
            metrics = torch.zeros(METRICS, len(dl), device = self.device)
        
            # Iterator
            progress = tqdm(enumerate(dl), total = len(dl), leave = True, bar_format = '{l_bar}{bar:15}{r_bar}{bar:-10b}')
            for batch_index, batch_tuple in progress:

                """
                ========================================================================================
                Prepare Data
                ========================================================================================
                """
                # Get MT and rCT
                # real1: MR; real2: rCT
                (images_t, labels_t, mask_t) = batch_tuple
                real1_g = images_t.to(self.device)
                real2_g = labels_t.to(self.device)
                mask_g = mask_t.to(self.device)

                # Z-Score Normalization
                real1_g -= real1_g.mean()
                real1_g /= real1_g.std()
                # Linear Sacling to [-1, 1]
                real1_g -= real1_g.min()
                real1_g /= real1_g.max()
                real1_g = (real1_g * 2) - 1

                # Linear Sacling to [-1, 1]
                real2_g = (real2_g * 2) - 1

                # Get sCT from Generator
                # fake2: sCT
                fake2_g = self.model(real1_g)

                """
                ========================================================================================
                Metrics
                ========================================================================================
                """
                # Reconstruction
                real2_g = ((real2_g + 1) * 2000) - 1000
                fake2_g = ((fake2_g + 1) * 2000) - 1000

                # MAE
                mae = Loss.get_mae(fake2_g, real2_g)

                # Head MAE
                head = Loss.get_head(fake2_g, real2_g, mask_g)

                # PSNR
                psnr = Loss.get_psnr(fake2_g, real2_g)

                # SSIM
                ssim = Loss.get_ssim(fake2_g, real2_g)

                # Save Metrics
                metrics[METRICS_MAE, batch_index] = mae
                metrics[METRICS_HEAD, batch_index] = head
                metrics[METRICS_PSNR, batch_index] = psnr
                metrics[METRICS_SSIM, batch_index] = ssim

                if mode == 'val':
                    progress.set_description('Evaluating Validation Set')
                else:
                    progress.set_description('Evaluating Testing Set')

        return metrics.to('cpu')

    """
    ================================================================================================
    Print Metrics' Mean and STD
    ================================================================================================
    """ 
    def print_result(self, metrics_t: Tensor) -> None:
        
        # Torch Tensor to Numpy Array
        metrics_a = metrics_t.detach().numpy()

        # Print Result
        space = "{: <15}\t{: <15.2f}\t{: <15.2f}"
        print()
        print(space.format('MAE', metrics_a[METRICS_MAE].mean(), metrics_a[METRICS_MAE].std()))
        print(space.format('Head', metrics_a[METRICS_HEAD].mean(), metrics_a[METRICS_HEAD].std()))
        print(space.format('PSNR', metrics_a[METRICS_PSNR].mean(), metrics_a[METRICS_PSNR].std()))
        print(space.format('SSIM', metrics_a[METRICS_SSIM].mean(), metrics_a[METRICS_SSIM].std()))
        print()

        return

    """
    ================================================================================================
    Save Image
    ================================================================================================
    """ 
    def save_images(self, mode: str) -> None:

        with torch.no_grad():

            # Model: Validation State
            self.model.eval()
        
            # Get Writer
            writer = getattr(self, mode + '_writer')

            # Get Data Loader
            dl = getattr(self, mode + '_dl')
                
            for i in range(3):

                index = random.randint(0, len(dl.dataset) - 1)

                # Get MT and rCT
                # real1: MR; real2: rCT; mask: Head Region
                (real1_t, real2_t, mask_t) = dl.dataset[index]
                real1_g = real1_t.to(self.device).unsqueeze(0)
                real2_g = real2_t.to(self.device).unsqueeze(0)

                # Z-Score Normalization
                real1_g -= real1_g.mean()
                real1_g /= real1_g.std()
                # Linear Sacling to [-1, 1]
                real1_g -= real1_g.min()
                real1_g /= real1_g.max()
                real1_g = (real1_g * 2) - 1

                # Linear Sacling to [-1, 1]
                real2_g = (real2_g * 2) - 1

                # Get sCT from Generator
                # fake2: sCT
                fake2_g = self.model(real1_g)

                # Torch Tensor to Numpy Array
                real1_a = real1_g.to('cpu').detach().numpy()[0]
                real2_a = real2_g.to('cpu').detach().numpy()[0]
                fake2_a = fake2_g.to('cpu').detach().numpy()[0]
                mask_a = mask_t.numpy()

                # Linear Sacling to [0, 1]
                real1_a -= real1_a.min()
                real1_a /= real1_a.max()

                # Linear Sacling to [0, 1]
                real2_a = (real2_a + 1) / 2
                fake2_a = (fake2_a + 1) / 2

                # Remove Background
                fake2_a = np.where(mask_a, fake2_a, 0)

                # Save Image to Tensorboard
                writer.add_image(self.time + '/' + str(i + 1) + 'MR', real1_a, dataformats = 'CHW')
                writer.add_image(self.time + '/' + str(i + 1) + 'rCT', real2_a, dataformats = 'CHW')
                writer.add_image(self.time + '/' + str(i + 1) + 'sCT', fake2_a, dataformats = 'CHW')

                """
                ============================================================================================
                Image: Difference Map
                ============================================================================================
                """
                # Reconstruction
                real2_a = (real2_a * 4000) - 1000
                fake2_a = (fake2_a * 4000) - 1000

                # Color Map
                colormap = LinearSegmentedColormap.from_list('colormap', [(1, 1, 1), (0, 0, 1), (1, 0, 0)])

                # Difference
                diff = np.abs(real2_a[0] - fake2_a[0])

                # Difference Map + Colorbar
                fig = plt.figure(figsize = (5, 5))
                plt.imshow(diff, cmap = colormap, vmin = 0, vmax = 2000, aspect = 'equal')
                plt.colorbar()

                # Save Image to Tensorboard
                writer.add_figure(self.time + '/' + str(i + 1) + 'Diff', fig)

                # Refresh Tensorboard Writer
                writer.flush()

        return
    

"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    eva = Evaluate()
    eva.main()