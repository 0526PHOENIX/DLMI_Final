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
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Unet import Unet, Pretrain
from Loss import get_pix_loss, get_gdl_loss, get_sim_loss
from Loss import get_mae, get_head, get_skull, get_dice
from Loss import get_psnr, get_ssim
from Dataset import Data


"""
====================================================================================================
Global Constant
====================================================================================================
"""
MAX = 10000000
STRIDE = 5
BATCH = 8
EPOCH = 2000
LR = 1e-6

PRETRAIN = True

METRICS = 7
METRICS_LOSS = 0
METRICS_MAE = 1
METRICS_HEAD = 2
METRICS_SKULL = 3
METRICS_DICE = 4
METRICS_PSNR = 5
METRICS_SSIM = 6

# Pixel-Wise Loss
LAMBDA_1 = 10
# Gradient Difference Loss
LAMBDA_2 = 3
# Similarity Loss
LAMBDA_3 = 1

DATA_PATH = ""
MODEL_PATH = ""
RESULTS_PATH = ""


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
    def __init__(self):
        
        # Training Device: CPU(cpu) or GPU(cuda)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('\n' + 'Training on: ' + str(self.device))

        # Model and Optimizer
        self.initialization()

        return

    """
    ================================================================================================
    Model and Optimizer
    ================================================================================================
    """
    def initialization(self):

        # Model: Unet
        if PRETRAIN:
            self.model = Pretrain().to(self.device)

        else:
            self.model = Unet().to(self.device)
        
        # Optimizer: Adam
        self.opt = Adam(self.model.parameters(), lr = LR)

        return

    """
    ================================================================================================
    TensorBorad
    ================================================================================================
    """
    def init_tensorboard(self):

        # Metrics Filepath
        log_dir = os.path.join(RESULTS_PATH, 'Metrics', self.time)

        # Tensorboard Writer
        self.train_writer = SummaryWriter(log_dir + '/Train')
        self.val_writer = SummaryWriter(log_dir + '/Val')

        return

    """
    ================================================================================================
    Data Loader
    ================================================================================================
    """
    def init_dl(self):

        # Training
        train_ds = Data(root = DATA_PATH, mode = 'Train')
        train_dl = DataLoader(train_ds, batch_size = BATCH, shuffle = True, drop_last = False)

        # Training Sample Index
        self.train_index = random.randint(0, len(train_ds) - 1)

        # Validation
        val_ds = Data(root = DATA_PATH, mode = 'Val')
        val_dl = DataLoader(val_ds, batch_size = BATCH, shuffle = True, drop_last = False)

        # Validation Sample Index
        self.val_index = random.randint(0, len(val_ds) - 1)

        return train_dl, val_dl
    
    """
    ================================================================================================
    Load Model Parameter and Hyperparameter
    ================================================================================================
    """
    def load_model(self):

        # Check Filepath
        if os.path.isfile(MODEL_PATH):

            # Get Checkpoint Information
            checkpoint = torch.load(MODEL_PATH)

            # Training Timestamp
            self.time = checkpoint['time']
            print('\n' + 'Continued From: ' +  self.time)

            # Model: Generator and Discriminator
            self.model.load_state_dict(checkpoint['model_state'])
            
            # Optimizer: Adam
            self.opt.load_state_dict(checkpoint['opt_state'])

            # Tensorboard Writer
            log_dir = os.path.join(RESULTS_PATH, 'Metrics', checkpoint['time'])
            self.train_writer = SummaryWriter(log_dir + '/Train')
            self.val_writer = SummaryWriter(log_dir + '/Val')

            # Dataset Sample Index
            self.train_index = checkpoint['train_index']
            self.val_index = checkpoint['val_index']

            # Begin Point
            if checkpoint['epoch'] < EPOCH:
                self.begin = checkpoint['epoch'] + 1
            else:
                self.begin = 1
            print('\n' + 'Start From Epoch: ' + str(self.begin))

            return checkpoint['score']
        
        else:
            
            # Training Timestamp
            self.time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
            print('\n' + 'Start From: ' + self.time)
            
            # Tensorboard
            self.init_tensorboard()

            # Save Hyperparameters
            self.save_hyper()

            # Begin Point
            self.begin = 1
        
            return MAX
    """
    ================================================================================================
    Main Training Function
    ================================================================================================
    """
    def main(self):

        # Data Loader
        train_dl, val_dl = self.init_dl()

        # Get Checkpoint
        best_score = self.load_model()

        # Main Training and Validation Loop
        for epoch_index in range(self.begin, EPOCH + 1):
            
            """
            ========================================================================================
            Training
            ========================================================================================
            """
            # Get Training Metrics
            print('\n' + 'Training: ')
            metrics_train = self.training(epoch_index, train_dl)

            # Save Training Metrics
            score = self.save_metrics(epoch_index, 'train', metrics_train)
            self.save_images(epoch_index, 'train', train_dl)
            self.save_model(epoch_index, score, False)

            # Validation: Stride = 5
            if epoch_index == 1 or epoch_index % STRIDE == 0:

                """
                ====================================================================================
                Validation
                ====================================================================================
                """
                # Get Validation Metrics
                print('===========================================================================')
                print('\n' + 'Validation: ')
                metrics_val = self.validation(epoch_index, val_dl)

                # Save Validation Metrics
                score = self.save_metrics(epoch_index, 'val', metrics_val)
                self.save_images(epoch_index, 'val', val_dl)

                # Save Model
                if not math.isnan(score):
                    best_score = min(best_score, score)
                self.save_model(epoch_index, score, (best_score == score))

                print('===========================================================================')
        
        # Close Tensorboard Writer
        self.train_writer.close()
        self.val_writer.close()

        return

    """
    ================================================================================================
    Training Loop
    ================================================================================================
    """
    def training(self, epoch_index, train_dl):
        
        # Model: Training State
        self.model.train()

        # Buffer for Metrics
        metrics = torch.zeros(METRICS, len(train_dl), device = self.device)

        # Progress Bar
        space = "{:3}{:3}{:3}"
        progress = tqdm(enumerate(train_dl), total = len(train_dl), leave = True,
                        bar_format = '{l_bar}{bar:15}{r_bar}{bar:-10b}')
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
            # Refresh Optimizer's Gradient
            self.opt.zero_grad()

            # Pixelwise Loss
            loss_pix = get_pix_loss(fake2_g, real2_g)

            # Gradient Difference loss
            loss_gdl = get_gdl_loss(fake2_g, real2_g)           

            # Similarity loss
            loss_sim = get_sim_loss(fake2_g, real2_g)              

            # Total Loss
            loss = LAMBDA_1 * loss_pix + LAMBDA_2 * loss_gdl + LAMBDA_3 * loss_sim

            # Update Generator's Parameters
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
            mae = get_mae(fake2_g, real2_g)

            # Head MAE
            head = get_head(fake2_g, real2_g, mask_g)

            # Skull MAE
            skull = get_skull(fake2_g, real2_g)

            # Skull Dice
            dice = get_dice(fake2_g, real2_g)

            # PSNR
            psnr = get_psnr(fake2_g, real2_g)

            # SSIM
            ssim = get_ssim(fake2_g, real2_g)

            # Save Metrics
            metrics[METRICS_LOSS, batch_index] = loss.item()
            metrics[METRICS_MAE, batch_index] = mae
            metrics[METRICS_HEAD, batch_index] = head
            metrics[METRICS_SKULL, batch_index] = skull
            metrics[METRICS_DICE, batch_index] = dice
            metrics[METRICS_PSNR, batch_index] = psnr
            metrics[METRICS_SSIM, batch_index] = ssim

            # Progress Bar Information
            progress.set_description('Epoch [' + space.format(epoch_index, ' / ', EPOCH) + ']')
            progress.set_postfix(ordered_dict = {'Loss': loss.item(), 'MAE': mae,
                                                 'Head': head, 'Skull': skull})

        return metrics.to('cpu')

    """
    ================================================================================================
    Validation Loop
    ================================================================================================
    """
    def validation(self, epoch_index, val_dl):

        with torch.no_grad():

            # Model: Validation State
            self.model.eval() 

            # Buffer for Metrics
            metrics = torch.zeros(METRICS, len(val_dl), device = self.device)
        
            # Progress Bar
            space = "{:3}{:3}{:3}"
            progress = tqdm(enumerate(val_dl), total = len(val_dl), leave = True,
                            bar_format = '{l_bar}{bar:15}{r_bar}{bar:-10b}')
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
                loss_pix = get_pix_loss(fake2_g, real2_g)      

                # Gradient Difference loss
                loss_gdl = get_gdl_loss(fake2_g, real2_g)

                # Similarity loss
                loss_sim = get_sim_loss(fake2_g, real2_g)              

                # Total Loss
                loss = LAMBDA_1 * loss_pix + LAMBDA_2 * loss_gdl + LAMBDA_3 * loss_sim

                """
                ========================================================================================
                Metrics
                ========================================================================================
                """
                # Reconstruction
                real2_g = ((real2_g + 1) * 2000) - 1000
                fake2_g = ((fake2_g + 1) * 2000) - 1000

                # MAE
                mae = get_mae(fake2_g, real2_g)

                # Head MAE
                head = get_head(fake2_g, real2_g, mask_g)

                # Skull MAE
                skull = get_skull(fake2_g, real2_g)

                # Skull Dice
                dice = get_dice(fake2_g, real2_g)

                # PSNR
                psnr = get_psnr(fake2_g, real2_g)

                # SSIM
                ssim = get_ssim(fake2_g, real2_g)

                # Save Metrics
                metrics[METRICS_LOSS, batch_index] = loss.item()
                metrics[METRICS_MAE, batch_index] = mae
                metrics[METRICS_HEAD, batch_index] = head
                metrics[METRICS_SKULL, batch_index] = skull
                metrics[METRICS_DICE, batch_index] = dice
                metrics[METRICS_PSNR, batch_index] = psnr
                metrics[METRICS_SSIM, batch_index] = ssim

                # Progress Bar Information
                progress.set_description('Epoch [' + space.format(epoch_index, ' / ', EPOCH) + ']')
                progress.set_postfix(ordered_dict = {'Loss': loss.item(), 'MAE': mae,
                                                     'Head': head, 'Skull': skull})

            return metrics.to('cpu')
        
    """
    ================================================================================================
    Save Hyperparameter: Batch Size, Epoch, Learning Rate
    ================================================================================================
    """
    def save_hyper(self):

        path = os.path.join(RESULTS_PATH, 'Metrics', self.time, 'Hyper.txt')

        with open(path, 'w') as f:

            print('Model:', 'Unet', file = f)
            print('Batch Size:', BATCH, file = f)
            print('Epoch:', EPOCH, file = f)
            print('Learning Rate:', LR, file = f)
            print('Pretrain:', PRETRAIN, file = f)
            print('Pix Loss Lamda:', LAMBDA_1, file = f)
            print('GDL Loss Lamda:', LAMBDA_2, file = f)
            print('Sim Loss Lamda:', LAMBDA_3, file = f)

        return
    
    """
    ================================================================================================
    Save Metrics for Whole Epoch
    ================================================================================================
    """ 
    def save_metrics(self, epoch_index, mode, metrics_t):

        # Get Writer
        writer = getattr(self, mode + '_writer')

        # Torch Tensor to Numpy Array
        metrics_a = metrics_t.detach().numpy().mean(axis = 1)

        # Create Dictionary
        metrics_dict = {}
        metrics_dict['Loss/LOSS'] = metrics_a[METRICS_LOSS]
        metrics_dict['Metrics/MAE'] = metrics_a[METRICS_MAE]
        metrics_dict['Metrics/MAE_Head'] = metrics_a[METRICS_HEAD]
        metrics_dict['Metrics/MAE_Skull'] = metrics_a[METRICS_SKULL]
        metrics_dict['Metrics/DICE_Skull'] = metrics_a[METRICS_DICE]
        metrics_dict['Metrics/PSNR'] = metrics_a[METRICS_PSNR]
        metrics_dict['Metrics/SSIM'] = metrics_a[METRICS_SSIM]

        # Save Metrics
        for key, value in metrics_dict.items():
            
            writer.add_scalar(key, value.item(), epoch_index)
        
        # Refresh Tensorboard Writer
        writer.flush()

        return metrics_dict['Metrics/MAE_Skull']

    """
    ================================================================================================
    Save Image
    ================================================================================================
    """ 
    def save_images(self, epoch_index, mode, dataloader):

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

            # Get Sample Index
            index = getattr(self, mode + '_index')

            # Get MT and rCT
            # real1: MR; real2: rCT; mask: Head Region
            (real1_t, real2_t, mask_t) = dataloader.dataset[index]
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
    def save_model(self, epoch_index, score, is_best = False):

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
        model_path = os.path.join(RESULTS_PATH, 'Model', self.time + '.pt')
        torch.save(state, model_path)

        # Save Best Model
        if is_best:
            best_path = os.path.join(RESULTS_PATH, 'Model', self.time + '.best.pt')
            torch.save(state, best_path)


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    Training().main()