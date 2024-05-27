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
    data = ""
    result = ""
    eva = ""
    weight = os.path.join("", time + '.pt')

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