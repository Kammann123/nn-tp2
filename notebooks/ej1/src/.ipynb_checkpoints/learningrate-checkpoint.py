"""
    @file   learningrate.py
    @desc   Contains classes used as Learning Rate Schedulers for dynamically changing the learning rate in training
    @author Joaquin Oscar Gaytan and Lucas Agustin Kammann
"""

# Third-party modules of Python
import numpy as np


class TimeBasedDecay:
    """ Computes the time based decay for the dynamic learing rate evolution
        throughout the training process.
    """
    
    def __init__(self, lr0: float, decay_rate : float):
        """ Creates a TimeBasedDecay instance
            @param lr0        Initial learning rate
            @param decay_rate Decaying rate of the learning rate
        """
        self.lr0 = lr0
        self.decay_rate = decay_rate
    
    def __call__(self, epoch : int):
        """ Compute the current value of learning rate
            @param epoch Iteration of the training process
        """
        return self.lr0 / ( 1 + self.decay_rate * epoch )


class StepDecay:
    """ Computes the step decay for the dynamic learning rate evolution
        throughout the training process.
    """
    
    def __init__(self, lr0: float, drop_rate: float, epochs_drop: int):
        """ Creates a StepDecay instance
            @param lr0         Initial learning rate
            @param drop_rate   Dropping rate of the learning rate
            @param epochs_drop Decaying rate of the learning rate
        """
        self.lr0 = lr0
        self.drop_rate = drop_rate
        self.epochs_drop = epochs_drop
    
    def __call__(self, epoch : float):
        """ Compute the current value of learning rate
            @param epoch Iteration of the training process
        """
        return self.lr0 * self.drop_rate ** np.floor( epoch / float(self.epochs_drop) )


class ExponentialDecay:
    """ Computes the exponential decay for the dynamic learning rate evolution
        throughout the training process.
    """

    def __init__(self, lr0: float, decay_rate: float, decay_steps : int = 1):
        """ Creates an ExponentialDecay instance
            @param lr0         Initial learning rate
            @param decay_rate  Decaying rate of the learning rate
            @param decay_steps Normalization of the epochs
        """
        self.lr0 = lr0
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def __call__(self, epoch : float):
        """ Compute the current value of learning rate
            @param epoch Iteration of the training process
        """
        return self.lr0 * np.exp(-self.decay_rate * (epoch / self.decay_steps))