"""
    @file   helper.py
    @desc   Contains general functions and classes
    @author Joaquin Oscar Gaytan and Lucas Agustin Kammann
"""

# Third-party modules of Python
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import datetime


def plot_kfold_metrics(train_maes, valid_maes, test_maes):
    """ Plot the performance of the model in each iteration in the train, valid and test sets.
        @param train_maes Metrics for each iteration in the train set
        @param valid_maes Metrics for each iteration in the valid set
        @param test_maes  Metrics for each iteration in the test set
    """
    # Create the plots
    plt.subplots(1, 1, figsize=(15, 8))
    
    # Create the k variable
    k = np.linspace(1, len(train_maes), len(train_maes))
    
    # Plot metrics in the train, valid and test sets
    plt.plot(k, train_maes, label='Train metric', color='blue', marker='o')
    plt.plot(k, valid_maes, label='Valid metric', color='red', marker='o')
    plt.plot(k, test_maes, label='Test metric', color='green', marker='o')
    
    # Add information and format the plot
    plt.title(f'K-Folding Validation (K={len(train_maes)})', fontsize=15)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.legend()
    plt.grid()
    plt.show()


def get_outliers(data, var):
    """ Get the outliers of the DataFrame column
        @param data Pandas DataFrame
        @param var  Name of the column of the DataFrame to extract outliers
        @return Array of outliers
    """
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
    """ Remove outliers from the DataFrame colum
        @param data Pandas DataFrame
        @param var  Name of the column of the DataFrame to extract outliers
    """
    outliers = get_outliers(data, var)
    for outlier in outliers:
        data[var].replace(outlier, np.nan, inplace=True)


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
