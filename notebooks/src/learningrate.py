"""
    @file learningrate.py
    @author Joaquin Oscar Gaytan and Lucas Agustin Kammann
"""

# Third-party modules of Python
import numpy as np


class StepDecay:
    """ Computes the step decay for the dynamic learning rate evolution
        throughout the training process.
    """
    
    def __init__(self, lr0: float, drop: float, epochs_drop: int):
        self.lr0 = lr0
        self.drop = drop
        self.epochs_drop = epochs_drop
    
    def __call__(self, epoch, lr):
        return self.lr0 * self.drop ** np.floor(epoch / float(self.epochs_drop))
    
class ExponentialDecay:
    """ Computes the exponential decay for the dynamic learning rate evolution
        throughout the training process.
    """

    def __init__(self, lr0: float, k: float):
        self.lr0 = lr0
        self.k = k

    def __call__(self, epoch, lr):
        return self.lr0 * np.exp(-self.k * epoch)