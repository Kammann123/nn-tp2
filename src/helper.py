"""
    @file helper.py
    @author Joaquin Oscar Gaytan and Lucas Agustin Kammann
"""

# Third-party modules of Python
import matplotlib.pyplot as plt
import numpy as np


def mean_absolute_error(true_results, predicted_results):
    """ Compute absolute and mean absolute error for each result comparing the predicted value
        with the true value throughout the dataset.
        @param true_results Array or list with the true annotated results
        @param predicted_results Array or list with the predicted results
        @return Tuple containing evolution of error throughout the dataset 
                => (absolute error, mean absolute error)
    """
    # Compute the error
    error = true_results - predicted_results
    # Compute the absolute error
    absolute_error = np.abs(error)
    # Compute the accumulative the error
    accumulative_error = np.cumsum(absolute_error)
    # Compute the mean absolute error through the samples
    mean_absolute_error = accumulative_error / np.array(range(1, 1 + len(accumulative_error)))
    return (absolute_error, mean_absolute_error)

def plot_linear_regression_history(history):
    """ Plots the history obtained from the training and validation process using the Keras algorithms.
        @param history History containing loss, validation loss and learning rate
    """
    # Create the layout and plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15))

    # Retrieve the fields, if existing
    loss = history.history['loss']
    val_loss = history.history['loss']
    lr = history.history['lr']

    # Create the loss plot
    ax1.plot(loss, label='Training Loss')
    ax1.plot(val_loss, label='Validation Loss')
    ax1.set_ylabel('Loss', fontsize=15)
    ax1.set_xlabel('n', fontsize=15)
    ax1.legend(fontsize=15)
    ax1.grid()

    # Create the learning rate plot
    ax2.plot(lr)
    ax2.set_ylabel('Learning Rate', fontsize=15)
    ax2.set_xlabel('n', fontsize=15)
    ax2.grid()

    # Show the plots
    plt.show()
    
def plot_linear_regression_result(true_results, predicted_results, result_label=''):
    """ Plots the real or true results for each case in the dataset and compares it with the 
        predicted result. It also plots the error for each case and the accumulative mean or 
        absolute error. 
        @param true_results Array or list with the true annotated results
        @param predicted_results Array or list with the predicted results
        @param result_label String label for the output of the linear regression
    """
    
    # Compute the errors
    abs_error, mean_abs_error = mean_absolute_error(true_results, predicted_results)
    
    # Create plots and layout
    fig, ax = plt.subplots(2, 1, figsize=(18, 12))

    # Plot the real and the prediction
    ax[0].plot(true_results, marker='o', label='Real')
    ax[0].plot(predicted_results, marker='o', label='Predicción')
    ax[0].set_xlabel('$n$', fontsize=15)
    ax[0].set_ylabel(result_label, fontsize=15)
    ax[0].legend(fontsize=12)
    ax[0].grid()

    # Plot the absolute error and the evolution of the mean absolute error throughout the dataset
    ax[1].plot(abs_error, label='Error absoluto')
    ax[1].plot(mean_abs_error, label='Error absolute medio')
    ax[1].legend(fontsize=12)
    ax[1].grid()

    # Show the graph
    plt.show()