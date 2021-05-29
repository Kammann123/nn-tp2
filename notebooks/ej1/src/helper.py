# Importing matplotlib modules
import matplotlib.pyplot as plt

# Importing pandas modules
import pandas as pd

# Importing numpy modules
import numpy as np

# Importing seaborn modules
import seaborn as sns

# Third-party modules of Python
import tensorflow.keras as keras
import tensorflow as tf
import datetime

def analyze_variable(data, var):
    # Create grid for figures
    fig, axs = plt.subplots(2, 2, figsize=(15, 13))
    
    # Plot 
    sns.histplot(data=data[var], kde=True, ax=axs[0][0], stat='density')
    axs[0][0].set_title('Distribución')
    
    sns.boxplot(data=data[var], ax=axs[0][1])
    axs[0][1].set_title('Boxplot')
    
    sns.histplot(data=data[var][data['Outcome'] == 0], kde=True, ax=axs[1][0], stat='density')
    axs[1][0].set_title('Distribución si no posee diabetes')
    
    sns.histplot(data=data[var][data['Outcome'] == 1], kde=True, ax=axs[1][1], stat='density')
    axs[1][1].set_title('Distribución si posee diabetes')
    
    axs[1][1].set_ylim(axs[1][0].get_ylim())
    
    # Show
    plt.show()
    
def get_outliers(data, var):
    # Usa criterio de "Outlier Leve"
    # extraído de https://es.wikipedia.org/wiki/Valor_at%C3%ADpico 
    q1 = data[var].quantile(0.25)
    q3 = data[var].quantile(0.75)
    iqr = q3 - q1
    mean = data[var].mean()
    ret = []
    for value in data[var]:
        if value < (q1 - 1.5 * iqr) or value > (q3 + 1.5 * iqr):
            ret.append(value)
    return ret

def remove_outliers(data, var): 
    outliers = get_outliers(data, var)
    for outlier in outliers:
        data[var].replace(outlier, np.nan, inplace=True)
        
        """
    @file   helper.py
    @desc   Contains general functions and classes
    @author Joaquin Oscar Gaytan and Lucas Agustin Kammann
"""



class LRTensorBoardLogger:
    """ Callable instance used to wrap a learning rate scheduler and log learning rate values 
        throughout the training process onto the TensorBoard platform.
    """
    
    def __init__(self, log_dir, schedule):
        """ Create a learning rate schedule that logs data onto TensorBoard.
            @param log_dir Logging directory for TensorBoard files
            @param schedule Function used to define the scheduling pattern for dynamic learning rate
        """
        
        # Save parameters as internal members
        self.log_dir = log_dir
        self.schedule = schedule
        
        # Create a file writer for TensorBoard logs
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.file_writer.set_as_default()
    
    def __call__(self, epoch):
        """ Compute the learning rate and logs it onto TensorBoard.
            @param epoch Current training epoch
            @return lr Learning rate
        """
        # Compute the new dynamic learning rate, log in onto TensorBoard and
        # return the result for the training process
        learning_rate = self.schedule(epoch)
        tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
        return learning_rate


def tensorboard_log(log_dir, tag, data):
    """ Log a scalar, a set of data or a time series in TensorBoard, by creating the proper log file
        in the logging directory, using the given tag and data.
        @param log_dir Logging directory where the TensorBoard file is created
        @param tag Tag used to group type of data or plots
        @param data Data to plot
    """
    # Create a file writer for TensorBoard logs
    file_writer = tf.summary.create_file_writer(log_dir)
    file_writer.set_as_default()

    # Send to TensorBoard both results
    for i in range(len(data)):
        tf.summary.scalar(tag, data=data[i], step=i)
        file_writer.flush()

        