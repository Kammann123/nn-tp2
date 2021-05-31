"""
    @file   helper.py
    @desc   Contains specific functions and classes to be used in the LR problem
    @author Joaquin Oscar Gaytan and Lucas Agustin Kammann
"""

# Third-party modules of Python
import sklearn.preprocessing as preprocessing
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import datetime
from sklearn.metrics import confusion_matrix, fbeta_score, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
from src.helper import remove_outliers
from sklearn import model_selection, preprocessing
from sklearn.model_selection import KFold

# Project modules of Python
import src.learningrate as learningrate
import src.helper as helper

    

def create_model(input_shape=8, l1=0, l2=0, dropout=0):
    
    # Define regularizer
    if l1 != 0 and l2 != 0:
        kernel_regularizer = keras.regularizers.l1_l2(l1=l1, l2=l2)
    elif l1 != 0:
        kernel_regularizer = keras.regularizers.l1(l1)
    elif l2 != 0:
        kernel_regularizer = keras.regularizers.l2(l2)
    else:
        kernel_regularizer = None
        
    # Define model
    model = keras.Sequential()
    model.add(keras.layers.Dense(1, input_shape=(input_shape,),
                                 activation='sigmoid',
                                 use_bias=True,
                                 kernel_regularizer=kernel_regularizer))
    if dropout != 0:
        model.add(keras.layers.Dropout(dropout))
        
    return model
        
    
    
def run_model_with_kfold(df,
                          test_size=0.2,
                          folds=10,
                          random_state=5,
                          *args,
                          **kwargs
                         ):
    
    
    # Filtering Glucose values
    df['Glucose'].replace(0, np.nan, inplace=True)

    # Filtering Blood Pressure values
    df['BloodPressure'].replace(0, np.nan, inplace=True)

    # Filtering Skin Thickness values
    df['SkinThickness'].replace(0, np.nan, inplace=True)

    # Filtering Insulin values
    df['Insulin'].replace(0, np.nan, inplace=True)

    # Filtering Body Mass Index values
    df['BMI'].replace(0, np.nan, inplace=True)
    
    x_labels = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age']
    y_labels = ['Outcome']

    for column in x_labels:
        remove_outliers(df, column)
        
    # Define input and output variables for the model
    df_x = df[x_labels]
    df_y = df[y_labels]
    
    # Split the dataset into train_valid and test
    x_train_valid, x_test, y_train_valid, y_test = model_selection.train_test_split(df_x, df_y, test_size=0.2, random_state=random_state, shuffle=True)

    # Compute mean for train-valid subset
    train_means = x_train_valid.mean().to_numpy()

    # Replacing nan values of the test dataset with training mean values
    for index, column in enumerate(x_test.columns):
        x_test.loc[:,column].replace(np.nan, train_means[index], inplace=True)
        
    # Init KFold
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    
    # Create arrays for metrics
    train_dict = {'auc' : [], 'specificity' : [], 'sensitivity' : [], 'ppv' : [], 'npv' : []}
    valid_dict = {'auc' : [], 'specificity' : [], 'sensitivity' : [], 'ppv' : [], 'npv' : []}    
    test_dict = {'auc' : [], 'specificity' : [], 'sensitivity' : [], 'ppv' : [], 'npv' : []}
    
    for train_index, valid_index in kf.split(x_train_valid):
        # Get train and valid splits
        x_train, y_train = x_train_valid.iloc[train_index], y_train_valid.iloc[train_index]
        x_valid, y_valid = x_train_valid.iloc[valid_index], y_train_valid.iloc[valid_index]
        
        # Compute the mean of training
        train_means = x_train.mean().to_numpy()

        # Replacing nan values of the train dataset with training mean values
        for index, column in enumerate(x_train.columns):
            x_train.loc[:,column].replace(np.nan, train_means[index], inplace=True)

        # Replacing nan values of the test dataset with training mean values
        for index, column in enumerate(x_valid.columns):
            x_valid.loc[:,column].replace(np.nan, train_means[index], inplace=True)
            
        eval_train, eval_valid, eval_test = run_model(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, x_test=x_test, y_test=y_test, *args, **kwargs)
        
        # Append scores for train, valid and test
        train_dict['auc'].append(eval_train['auc'])
        train_dict['specificity'].append(eval_train['specificity'])
        train_dict['sensitivity'].append(eval_train['sensitivity'])
        train_dict['ppv'].append(eval_train['ppv'])
        train_dict['npv'].append(eval_train['npv'])

        valid_dict['auc'].append(eval_valid['auc'])
        valid_dict['specificity'].append(eval_valid['specificity'])
        valid_dict['sensitivity'].append(eval_valid['sensitivity'])
        valid_dict['ppv'].append(eval_valid['ppv'])
        valid_dict['npv'].append(eval_valid['npv'])

        test_dict['auc'].append(eval_test['auc'])
        test_dict['specificity'].append(eval_test['specificity'])
        test_dict['sensitivity'].append(eval_test['sensitivity'])
        test_dict['ppv'].append(eval_test['ppv'])
        test_dict['npv'].append(eval_test['npv'])
        
    train_dict['auc'] = np.nanmean(train_dict['auc'])
    train_dict['specificity'] = np.nanmean(train_dict['specificity'])
    train_dict['sensitivity'] = np.nanmean(train_dict['sensitivity'])
    train_dict['ppv'] = np.nanmean(train_dict['ppv'])
    train_dict['npv'] = np.nanmean(train_dict['npv'])

    valid_dict['auc'] = np.nanmean(valid_dict['auc'])
    valid_dict['specificity'] = np.nanmean(valid_dict['specificity'])
    valid_dict['sensitivity'] = np.nanmean(valid_dict['sensitivity'])
    valid_dict['ppv'] = np.nanmean(valid_dict['ppv'])
    valid_dict['npv'] = np.nanmean(valid_dict['npv'])

    test_dict['auc'] = np.nanmean(test_dict['auc'])
    test_dict['specificity'] = np.nanmean(test_dict['specificity'])
    test_dict['sensitivity'] = np.nanmean(test_dict['sensitivity'])
    test_dict['ppv'] = np.nanmean(test_dict['ppv'])
    test_dict['npv'] = np.nanmean(test_dict['npv'])

    
    return train_dict, valid_dict, test_dict
    
    
def run_model(x_train, y_train, x_valid, y_valid, x_test, y_test,
              learning_rate,
              tag='untagged',
              degree=1,
              scheduler=None,
              decay_rate=0.1,
              drop_rate=0.5,
              epochs_drop=10,
              optimizer='sgd',
              momentum=0,
              rho=0,
              beta_1=0,
              beta_2=0,
              batch_size=32,
              epochs=100,
              patience=50,
              min_delta=10,
              tensorboard_on=True,
              checkpoints_on=True,
              summary_on=True,
              *args,
              **kwargs
             ):
    
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create the logging directory name
    log_dir = 'tb-logs/rl/' + tag + '/' + timestamp
    
    # Create the model checkpoint directory name
    checkpoint_dir = 'checkpoints/rl/' + tag + timestamp
    
    # Create the polynomial features preprocessor
    poly = preprocessing.PolynomialFeatures(degree=degree, include_bias=False)
    poly.fit(x_train)
    x_train = poly.transform(x_train)
    x_valid = poly.transform(x_valid)
    x_test = poly.transform(x_test)
    
    # Normalize the variables
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)
    
    # Create the model
    model = create_model(input_shape=poly.n_output_features_, *args, **kwargs)
    if summary_on:
        if tensorboard_on:
            print(f'Model logs at {log_dir}')
        print(f'Model checkpoints at {checkpoint_dir}')
        model.summary()
        
    # Create the optimizer
    if optimizer == 'adam':
        model_optimizer = keras.optimizers.Adam(beta_1=beta_1, beta_2=beta_2)
    elif optimizer == 'sgd':
        model_optimizer = keras.optimizers.SGD(momentum=momentum)
    elif optimizer == 'rmsprop':
        model_optimizer = keras.optimizers.RMSprop(momentum=momentum, rho=rho)
    else:
        raise ValueError('Unknown or unsupported optimizer, expected: adam, sgd or rmsprop')
        
    # Compile the model
    model.compile(optimizer=model_optimizer,
                  loss='binary_crossentropy',
                  metrics=['AUC']
                 )
    
    # Create the learning rate scheduler and callback
    if scheduler == 'exponential-decay':
        lr_scheduler = learningrate.ExponentialDecay(learning_rate, decay_rate)
    elif scheduler == 'time-decay':
        lr_scheduler = learningrate.TimeBasedDecay(learning_rate, decay_rate)
    elif scheduler == 'step-decay':
        lr_scheduler = learningrate.StepDecay(learning_rate, drop_rate, epochs_drop)
    else:
        lr_scheduler = lambda epoch: learning_rate
        
    
    lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)
    
    if checkpoints_on:
        mc_callback = keras.callbacks.ModelCheckpoint(checkpoint_dir + '.hdf5', monitor='val_auc', save_best_only=True, mode='max', verbose=0)
        
    # Create the tensorboard callback
    if tensorboard_on:
        tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                  histogram_freq=1, 
                                                  update_freq='epoch'
                                                 )
    # Create the early stopping callback
    es_callback = keras.callbacks.EarlyStopping(monitor='val_auc',
                                                mode='max',
                                                patience=patience,
                                                min_delta=min_delta,
                                                restore_best_weights=True
                                               )
    # Define callbacks used
    callbacks = [ es_callback, lr_callback ]
    if tensorboard_on:
        callbacks += [ tb_callback ]
    if checkpoints_on:
        callbacks += [ mc_callback ]
        
    # Train the neural network
    model.fit(x_train, y_train,
              validation_data=(x_valid, y_valid),
              epochs=epochs, 
              verbose=0, 
              shuffle=True, 
              batch_size=batch_size,
              callbacks=callbacks,
              use_multiprocessing=True
             )
    
    # Load the best model and evaluate the metric
    model = keras.models.load_model(checkpoint_dir + '.hdf5')
    
    eval_train, eval_valid, eval_test = helper.get_metrics(model, x_train, y_train, x_valid, y_valid, x_test, y_test,verbose=summary_on, f2_plot=summary_on);
        
    return eval_train, eval_valid, eval_test
    