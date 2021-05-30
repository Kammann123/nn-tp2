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
from sklearn.metrics import confusion_matrix, fbeta_score, roc_auc_score
import datetime
import matplotlib.pyplot as plt

def get_metrics(model, x_train, y_train, x_valid, y_valid, x_test, y_test, threshold=0.5, verbose=True, f2_plot=False):
    """
    Computes the following metrics:
        * AUC
        * Specifitiy
        * Sensitivity
        * PV+ (PPV)
        * PV- (NPV)
    """
    
    # Get prediction from model for each subset
    y_train_probs = model.predict(x=x_train)
    y_valid_probs = model.predict(x=x_valid)
    y_test_probs = model.predict(x=x_test)
    
    # Round predictions to class
    y_train_pred = round_threshold(y_train_probs, threshold)
    y_valid_pred = round_threshold(y_valid_probs, threshold)
    y_test_pred = round_threshold(y_test_probs, threshold)
    
    # Get TP, FP, TN, FN to compute metrics
    tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_true=y_train, y_pred=y_train_pred).ravel()
    tn_valid, fp_valid, fn_valid, tp_valid = confusion_matrix(y_true=y_valid, y_pred=y_valid_pred).ravel()
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_true=y_test, y_pred=y_test_pred).ravel()
    
    
    # Define dictionaries to return
    train_dict = {'auc' : 0, 'specificity' : 0, 'sensitivity' : 0, 'ppv' : 0, 'npv' : 0}
    valid_dict = {'auc' : 0, 'specificity' : 0, 'sensitivity' : 0, 'ppv' : 0, 'npv' : 0}
    test_dict = {'auc' : 0, 'specificity' : 0, 'sensitivity' : 0, 'ppv' : 0, 'npv' : 0}
    
    # Calculate AUC (Keras) - slower
    #eval_train = model.evaluate(x_train, y_train, verbose=0, return_dict=True)
    #eval_valid = model.evaluate(x_valid, y_valid, verbose=0, return_dict=True)
    #eval_test = model.evaluate(x_test, y_test, verbose=0, return_dict=True)

    #auc_train = eval_train['auc']
    #auc_valid= eval_valid['auc']
    #auc_test = eval_test['auc']
    
    # Calculate AUC (SKLearn) - faster
    auc_train_sk = roc_auc_score(y_true=y_train, y_score=y_train_probs)
    auc_valid_sk = roc_auc_score(y_true=y_valid, y_score=y_valid_probs)
    auc_test_sk = roc_auc_score(y_true=y_test, y_score=y_test_probs)
    
    train_dict['auc'] = auc_train_sk
    valid_dict['auc'] = auc_valid_sk
    test_dict['auc'] = auc_test_sk

    
    # Calculate positive predictive value
    ppv_train = tp_train / (fp_train + tp_train)
    ppv_valid = tp_valid / (fp_valid + tp_valid)
    ppv_test = tp_test / (fp_test + tp_test)
    
    train_dict['ppv'] = ppv_train
    valid_dict['ppv'] = ppv_valid
    test_dict['ppv'] = ppv_test
    
    # Calculate negative predicitve value
    npv_train = tn_train / (fn_train + tn_train)
    npv_valid = tn_valid / (fn_valid + tn_valid)
    npv_test = tn_test / (fn_test + tn_test)
    
    train_dict['npv'] = npv_train
    valid_dict['npv'] = npv_valid
    test_dict['npv'] = npv_test
    
    # Calculate sensitivity
    sensitivity_train = tp_train / (tp_train + fn_train)
    sensitivity_valid = tp_valid / (tp_valid + fn_valid)
    sensitivity_test = tp_test / (tp_test + fn_test)
    
    train_dict['sensitivity'] = sensitivity_train
    valid_dict['sensitivity'] = sensitivity_valid
    test_dict['sensitivity'] = sensitivity_test
    
    # Calculate specificity
    specificity_train = tn_train / (tn_train + fp_train)
    specificity_valid = tn_valid / (tn_valid + fp_valid)
    specificity_test = tn_test / (tn_test + fp_test)
    
    train_dict['specificity'] = specificity_train
    valid_dict['specificity'] = specificity_valid
    test_dict['specificity'] = specificity_test
    
    if verbose == True:
        thrld, f2_score, idx = f2_threshold_selection(y_valid_probs, y_valid, y_train_probs, y_train, steps=100, plot=f2_plot)
        best_threshold = thrld[idx]
        best_f2 = f2_score[idx]
        print('------------------- Main metric -------------------')
        #print(f'[AUC] Train: {auc_train:.4f} - Valid: {auc_valid:.4f} - Test: {auc_test:.4f}')
        print(f'[AUC] Train: {auc_train_sk:.4f} - Valid: {auc_valid_sk:.4f} - Test: {auc_test_sk:.4f}')
        print(f'---------------- Secondary metrics ----------------')
        print(f'[PPV] Train: {ppv_train:.4f} - Valid: {ppv_valid:.4f} - Test: {ppv_test:.4f}')
        print(f'[NPV] Train: {npv_train:.4f} - Valid: {npv_valid:.4f} - Test: {npv_test:.4f}')
        print(f'[SEN] Train: {sensitivity_train:.4f} - Valid: {sensitivity_valid:.4f} - Test: {sensitivity_test:.4f}')
        print(f'[SPE] Train: {specificity_train:.4f} - Valid: {specificity_valid:.4f} - Test: {specificity_test:.4f}')
        print('---------------- Confusion Matrix -----------------')
        print(f'Train: FP = {fp_train} - TP = {tp_train} - FN = {fn_train} - TN = {tn_train}')
        print(f'Valid: FP = {fp_valid} - TP = {tp_valid} - FN = {fn_valid} - TN = {tn_valid}')
        print(f'Test: FP = {fp_test} - TP = {tp_test} - FN = {fn_test} - TN = {tn_test}')
        print(f'--------------- Threshold Selection ---------------')
        print(f'[F2S] Best f2 score for valid is {best_f2:.4f} @ threhsold = {best_threshold:.4f}')
        
    
    return train_dict, valid_dict, test_dict

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

def f2_threshold_selection(y_probs_valid, y_true_valid, y_probs_train, y_true_train, steps=100, plot=True):
    # Thresholds and f2-score vectors
    thresholds = np.linspace(0, 1, steps)
    f2_score_valid = []
    f2_score_train = []
    
    for thld in thresholds:
        # Generate predictions with current threshold
        y_pred_valid = round_threshold(vector=y_probs_valid, threshold=thld)
        y_pred_train = round_threshold(vector=y_probs_train, threshold=thld)
        # Compute f2 score for that threshold and append
        score_valid = fbeta_score(y_true=y_true_valid, y_pred=y_pred_valid, beta=2)
        score_train = fbeta_score(y_true=y_true_train, y_pred=y_pred_train, beta=2)
        f2_score_valid.append(score_valid)
        f2_score_train.append(score_train)
    
    idx = np.argmax(f2_score_valid)
    if plot == True:
        plt.plot(thresholds, f2_score_valid, label='valid')
        plt.plot(thresholds, f2_score_train, label='train')
        plt.xlabel('Threshold')
        plt.ylabel('F2 score')
        plt.axvline(thresholds[idx], color='black', linestyle='--')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.grid(b=True)
        plt.title('Selecting best threshold')
        plt.legend()
        plt.show()
        
    return thresholds, f2_score_valid, idx

def round_threshold(vector, threshold=0.5):
    rounded_vector = []
    for element in vector:
        if element >= threshold:
            rounded_vector.append(1)
        else:
            rounded_vector.append(0)

    return np.array(rounded_vector)


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

        