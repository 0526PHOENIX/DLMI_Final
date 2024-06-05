"""
====================================================================================================
Package
====================================================================================================
"""
import os

from torch.utils.tensorboard import SummaryWriter

from Training import Training
from Evaluate import Evaluate


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    # Critical Info
    train = True
    time = ""

    # Filepath
    weight = os.path.join("", time + '.pt')

    if train: 

        """
        ============================================================================================
        Training
        ============================================================================================
        """
        # Model Weight Path
        weight = os.path.join("", time + '.pt')

        # Parameter
        params = {'epoch': 4000,
                  'batch': 8,
                  'lr': 1e-6,
                  'augment': True,
                  'depth': 5,
                  'bottle': 9,
                  'loss': [5, 7, 1]}
        
        # Training
        training = Training(**params, weight = weight)
        training.main()
    
    else:

        """
        ============================================================================================
        Evaluation
        ============================================================================================
        """
        # Training Timestamp
        times = os.listdir("")

        for time in times:

            # Model Depth and Bottleneck Length
            with open(os.path.join("", time, 'Hyper.txt')) as f:

                for line in f.readlines():

                    if 'Model Depth' in line:
                        depth = int(line.split(':')[1].strip())
                    elif 'Bottleneck Length' in line:
                        bottleneck = int(line.split(':')[1].strip())

            # Model Weight Path
            weight = os.path.join("", time + '.best.pt')

            # Parameter
            params = {'depth': depth,
                      'bottle': bottleneck}
            
            # Evaluation
            evaluation = Evaluate(**params, weight = weight)
            evaluation.main()
            print('===============================================================================')