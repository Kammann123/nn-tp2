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
    
    def __call__(self, epoch):
        return self.lr0 * self.drop ** np.floor(epoch / float(self.epochs_drop))


class ExponentialDecay:
    """ Computes the exponential decay for the dynamic learning rate evolution
        throughout the training process.
    """

    def __init__(self, lr0: float, decay_rate: float, decay_steps : int = 1):
        self.lr0 = lr0
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def __call__(self, epoch):
        return self.lr0 * np.exp(-self.decay_rate * np.floor(epoch / self.decay_steps))