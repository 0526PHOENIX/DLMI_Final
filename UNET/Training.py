"""
====================================================================================================
Package
====================================================================================================
"""
import os
import math
import datetime
import random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import warnings
warnings.filterwarnings('ignore')

import torch
from torch import Tensor
from torch.optim import Adam, RMSprop, SGD, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Utils import *


"""
====================================================================================================
Global Constant
====================================================================================================
"""
MAX = 10000000

METRICS = 5
METRICS_LOSS = 0
METRICS_MAE = 1
METRICS_HEAD = 2
METRICS_PSNR = 3
METRICS_SSIM = 4


"""
====================================================================================================
Training
====================================================================================================
"""
class Training():

    """
    ================================================================================================
    Critical Parameters
    ================================================================================================
    """
    def __init__(self,
                 epoch: int = 4000,
                 batch: int = 8,
                 lr: float = 1e-3,
                 depth: int = 5,
                 bottle: int = 9,
                 augment: bool = False,
                 loss: list[int] = [10, 3, 5],
                 data: str = "",
                 result: str = "",
                 weight: str = "",
                 *args,
                 **kwargs) -> None:
        
        # Training Device: CPU(cpu) or GPU(cuda)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        print('\n' + 'Training on: ' + str(self.device))

        # Total Epoch & Batch Size
        self.epoch = epoch
        self.batch = batch

        # Learning Rate
        self.lr = lr

        # Model
        self.depth = depth
        self.bottle = bottle

        # Dataset
        self.augment = augment

        # Loss Function Weight
        # labmda_1: PIX Loss; lambda_2: GDL Loss; lambda_3: SIM Loss
        (self.lambda_1, self.lambda_2, self.lambda_3) = loss

        # File Path
        self.data = data
        self.result = result
        self.weight = weight

        # Model and Optimizer
        self.initialization()

        return

    """
    ================================================================================================
    Model and Optimizer
    ================================================================================================
    """
    def initialization(self) -> None:

        # Model: Unet
        self.model = Unet(depth = self.depth, bottle = self.bottle).to(self.device)
        
        # Optimizer: Adam
        self.opt = Adam(self.model.parameters(), lr = self.lr)

        # Learning Rate cheduler
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.opt, T_max = 50, eta_min = 1e-10)

        return

    """
    ================================================================================================
    TensorBorad
    ================================================================================================
    """
    def init_tensorboard(self) -> None:

        # Metrics Filepath
        log_dir = os.path.join(self.result, 'Metrics', self.time)

        # Tensorboard Writer
        self.train_writer = SummaryWriter(log_dir + '/Train')
        self.val_writer = SummaryWriter(log_dir + '/Val')

        # Save Model Architecture
        dummy = torch.randn(1, 1, 256, 256).to(device = self.device)
        self.train_writer.add_graph(self.model, dummy)

        return

    """
    ================================================================================================
    Data Loader
    ================================================================================================
    """
    def init_dl(self) -> None:

        root = os.path.join(self.data, 'Data')

        # Training Dataset
        train_ds = Data(root = root, mode = 'Train', augment = self.augment)
        self.train_dl = DataLoader(train_ds, batch_size = self.batch, shuffle = True, drop_last = False,
                                   num_workers = 4, pin_memory = True)

        # Validation Dataset
        val_ds = Data(root = root, mode = 'Val', augment = False)
        self.val_dl = DataLoader(val_ds, batch_size = self.batch, shuffle = True, drop_last = False,
                                 num_workers = 4, pin_memory = True)

        # Training Sample Index
        self.train_index = random.randint(0, len(train_ds) - 1)

        # Validation Sample Index
        self.val_index = random.randint(0, len(val_ds) - 1)

        return
    
    """
    ================================================================================================
    Load Model Parameter and Hyperparameter
    ================================================================================================
    """
    def load_model(self) -> float:

        # Check Filepath
        if os.path.isfile(self.weight):

            # Get Checkpoint Information
            checkpoint = torch.load(self.weight)

            # Training Timestamp
            self.time = checkpoint['time']
            print('\n' + 'Continued From: ' +  self.time)

            # Model: Generator and Discriminator
            self.model.load_state_dict(checkpoint['model_state'])
            
            # Optimizer: Adam
            self.opt.load_state_dict(checkpoint['opt_state'])

            # Tensorboard Writer
            log_dir = os.path.join(self.result, 'Metrics', checkpoint['time'])
            self.train_writer = SummaryWriter(log_dir + '/Train')
            self.val_writer = SummaryWriter(log_dir + '/Val')

            # Dataset Sample Index
            self.train_index = checkpoint['train_index']
            self.val_index = checkpoint['val_index']

            # Begin Point
            if checkpoint['epoch'] < self.epoch:
                self.begin = checkpoint['epoch'] + 1
                print('\n' + 'Continued From Epoch: ' + str(self.begin))
            else:
                raise ValueError('Invalid Epoch Number, Would Destroy Saved Training Curve')

            return checkpoint['score']
        
        else:
            
            # Training Timestamp
            self.time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
            print('\n' + 'Start From: ' + self.time)
            
            # Tensorboard
            self.init_tensorboard()

            # Begin Point
            self.begin = 1
        
            return MAX

    """
    ================================================================================================
    Main Training Function
    ================================================================================================
    """
    def main(self) -> None:

        # Data Loader
        self.init_dl()

        # Get Checkpoint
        best_score = self.load_model()

        # Main Training and Validation Loop
        for epoch_index in range(self.begin, self.epoch + 1):
            
            """
            ========================================================================================
            Training
            ========================================================================================
            """
            # Get Training Metrics
            print('\n' + 'Training: ')
            metrics_train = self.training(epoch_index)

            # Save Training Metrics
            score = self.save_metrics(epoch_index, 'train', metrics_train)
            self.save_images(epoch_index, 'train')

            # Save Model
            self.save_model(epoch_index, score, False)

            # Validation: Stride = 10
            if epoch_index == 1 or epoch_index % 5 == 0:

                """
                ====================================================================================
                Validation
                ====================================================================================
                """
                # Get Validation Metrics
                print('===========================================================================')
                print('\n' + 'Validation: ')
                metrics_val = self.validation(epoch_index)

                # Save Validation Metrics
                score = self.save_metrics(epoch_index, 'val', metrics_val)
                self.save_images(epoch_index, 'val')

                # Save Model
                if not math.isnan(score):
                    best_score = min(best_score, score)
                self.save_model(epoch_index, score, (best_score == score))

                print('===========================================================================')

            # Learning Rate Scheduler
            self.scheduler.step(score)

            # Adaptive Loss Function Weight
            if epoch_index % 2000 == 0:
                self.lambda_1 *= 1.5
                self.lambda_2 *= 1.0
                self.lambda_3 *= 0.5
            
        # Close Tensorboard Writer
        self.train_writer.close()
        self.val_writer.close()

        return

    """
    ================================================================================================
    Training
    ================================================================================================
    """
    def training(self, epoch_index: int) -> Tensor:
        
        # Model: Training State
        self.model.train()

        return self.looping(epoch_index, 'train')

    """
    ================================================================================================
    Validation
    ================================================================================================
    """
    def validation(self, epoch_index: int) -> Tensor:

        with torch.no_grad():

            # Model: Validation State
            self.model.eval() 

            return self.looping(epoch_index, 'val')
        
    """
    ================================================================================================
    Looping
    ================================================================================================
    """      
    def looping(self, epoch_index: int, mode: str) -> Tensor:

        # Get Data Loader
        dl = getattr(self, mode + '_dl')

        # Buffer for Metrics
        metrics = torch.zeros(METRICS, len(dl), device = self.device)

        # Iterator
        progress = tqdm(enumerate(dl), total = len(dl), leave = True, bar_format = '{l_bar}{bar:15}{r_bar}{bar:-10b}')
            
        # Progress Bar
        space = "{:3}{:3}{:3}"
        for batch_index, batch_tuple in progress:

            """
            ========================================================================================
            Prepare Data
            ========================================================================================
            """
            # Get MT and rCT
            # real1: MR; real2: rCT; mask: Head Region
            (real1_t, real2_t, mask_t) = batch_tuple
            real1_g = real1_t.to(self.device)
            real2_g = real2_t.to(self.device)
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
            Unet
            ========================================================================================
            """
            # Pixelwise Loss
            loss_pix = Loss.get_pix_loss(fake2_g, real2_g)

            # Gradient Difference loss
            loss_gdl = Loss.get_gdl_loss(fake2_g, real2_g)

            # Similarity loss
            loss_sim = Loss.get_sim_loss(fake2_g, real2_g)           

            # Total Loss
            loss = (self.lambda_1*loss_pix) + (self.lambda_2*loss_gdl) + (self.lambda_3*loss_sim)

            # Gradient Descent
            if mode == 'train':

                # Update Generator's Parameters
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

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
            metrics[METRICS_LOSS, batch_index] = loss.item()
            metrics[METRICS_MAE, batch_index] = mae
            metrics[METRICS_HEAD, batch_index] = head
            metrics[METRICS_PSNR, batch_index] = psnr
            metrics[METRICS_SSIM, batch_index] = ssim

            # Progress Bar Information
            progress.set_description('Epoch [' + space.format(epoch_index, ' / ', self.epoch) + ']')
            progress.set_postfix(ordered_dict = {'MAE': mae, 'Head': head})

            # Release Memory
            del fake2_g
            del loss_pix, loss_gdl, loss_sim, loss
            del mae, head, psnr, ssim

        return metrics.to('cpu')
    
    """
    ================================================================================================
    Save Hyperparameter: Batch Size, Epoch, Learning Rate
    ================================================================================================
    """
    def save_hyper(self, epoch_index: int) -> None:

        path = os.path.join(self.result, 'Metrics', self.time, 'Hyper.txt')

        with open(path, 'w') as f:

            print('Model:', 'Unet', file = f)
            print('Epoch:', epoch_index, file = f)
            print('Batch:', self.batch, file = f)
            print('Learning Rate:', self.lr, file = f)
            print('Augmentation:', self.augment, file = f)
            print('Model Depth:', self.depth, file = f)
            print('Bottleneck Length:', self.bottle, file = f)
            print('Pix Loss Lamda:', self.lambda_1, file = f)
            print('GDL Loss Lamda:', self.lambda_2, file = f)
            print('Sim Loss Lamda:', self.lambda_3, file = f)

        return

    """
    ================================================================================================
    Save Metrics for Whole Epoch
    ================================================================================================
    """ 
    def save_metrics(self, epoch_index: int, mode: str, metrics_t: Tensor) -> float:

        # Get Writer
        writer = getattr(self, mode + '_writer')

        # Torch Tensor to Numpy Array
        metrics_a = metrics_t.detach().numpy().mean(axis = 1)

        # Create Dictionary
        metrics_dict = {}
        metrics_dict['Loss/LOSS'] = metrics_a[METRICS_LOSS]
        metrics_dict['Metrics/MAE'] = metrics_a[METRICS_MAE]
        metrics_dict['Metrics/MAE_Head'] = metrics_a[METRICS_HEAD]
        metrics_dict['Metrics/PSNR'] = metrics_a[METRICS_PSNR]
        metrics_dict['Metrics/SSIM'] = metrics_a[METRICS_SSIM]

        # Save Metrics
        for key, value in metrics_dict.items():
            
            writer.add_scalar(key, value, epoch_index)
        
        # Refresh Tensorboard Writer
        writer.flush()

        return metrics_dict['Metrics/MAE_Head']

    """
    ================================================================================================
    Save Image
    ================================================================================================
    """ 
    def save_images(self, epoch_index: int, mode: str) -> None:

        with torch.no_grad():

            """
            ============================================================================================
            Image: MR, rCT, sCT
            ============================================================================================
            """ 
            # Model: Validation State
            self.model.eval()
        
            # Get Writer
            writer = getattr(self, mode + '_writer')

            # Get Data Loader
            dl = getattr(self, mode + '_dl')

            # Get Sample Index
            index = getattr(self, mode + '_index')

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

            # Save Image
            writer.add_image(mode + '/MR', real1_a, epoch_index, dataformats = 'CHW')
            writer.add_image(mode + '/rCT', real2_a, epoch_index, dataformats = 'CHW')
            writer.add_image(mode + '/sCT', fake2_a, epoch_index, dataformats = 'CHW')

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

            # Save Image
            writer.add_figure(mode + '/Diff', fig, epoch_index)

            # Refresh Tensorboard Writer
            writer.flush()

        return

    """
    ================================================================================================
    Save Model
    ================================================================================================
    """ 
    def save_model(self, epoch_index: int, score: float, is_best: bool = False) -> None:

        # Time, Model State, Optimizer State
        # Ending Epoch, Best Score
        state = {
            'time': self.time,
            'model_state': self.model.state_dict(),
            'model_name': type(self.model).__name__,
            'opt_state': self.opt.state_dict(),
            'opt_name': type(self.opt).__name__,
            'train_index': self.train_index,
            'val_index': self.val_index,
            'epoch': epoch_index,
            'score': score,
        }

        # Save Model
        model_path = os.path.join(self.result, 'Model', self.time + '.pt')
        torch.save(state, model_path)

        if is_best:
            # Save Best Model
            best_path = os.path.join(self.result, 'Model', self.time + '.best.pt')
            torch.save(state, best_path)

        else:
            # Save Hyperparameters
            self.save_hyper(epoch_index)

        return


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    train = Training()
    train.main()