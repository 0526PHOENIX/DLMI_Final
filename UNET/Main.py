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
    time = "2024-05-27_23-35"

    # Filepath
    data = "C:/Users/user/Desktop/DLMI"
    result = "C:/Users/user/Desktop/DLMI/UNET/Result"
    eva = "C:/Users/user/Desktop/DLMI/UNET/Evaluate"
    weight = os.path.join("C:/Users/user/Desktop/DLMI/UNET/Result/Model", time + '.pt')

    if train: 

        # Training
        params = {'epoch': 4000,
                  'batch': 8,
                  'lr': 1e-6,
                  'augment': True,
                  'depth': 5,
                  'bottle': 12,
                  'loss': [10, 3, 5],
                  'data': data,
                  'result': result}
                        
        training = Training(**params, weight = weight)
        training.main()
    
    else:

        # Evaluation
        params = {'depth': 5,
                  'bottle': 12,
                  'data': data,
                  'eva': eva}
        
        evaluation = Evaluate(**params, weight = weight)
        evaluation.main()