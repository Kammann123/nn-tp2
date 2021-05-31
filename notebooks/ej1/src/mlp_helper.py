import tensorflow.keras as keras
import tensorflow as tf
import datetime

import src.learningrate as learningrate
import src.helper as helper
from src.helper import get_metrics

from src.helper import remove_outliers
from sklearn import model_selection, preprocessing
from sklearn.model_selection import KFold
import numpy as np


def create_model(hidden_layers=0, units_per_layer=0, hidden_layer_activation=None, regularizer=None,regularizer_lambda=1e-10, dropout_rate=0.0, use_batch_normalization=False):
    # Regularizers
    if regularizer == 'l1':
        kernel_regularizer = keras.regularizers.l1(regularizer_lambda)
    elif regularizer == 'l2':
        kernel_regularizer = keras.regularizers.l2(regularizer_lambda)
    else:
        kernel_regularizer = None
        
    # Create model
    model = keras.models.Sequential()
    
    # Create input layer
    model.add(keras.layers.InputLayer(input_shape=(8, )))
    
    for layer_index in range(hidden_layers):
        # Create dense layer
        model.add(keras.layers.Dense(units=units_per_layer, activation=hidden_layer_activation, kernel_initializer='random_normal', kernel_regularizer=kernel_regularizer))
        # Check for additional layers
        if dropout_rate:
            model.add(keras.layers.Dropout(dropout_rate))
        if use_batch_normalization:
            model.add(keras.layers.BatchNormalization())
    
    # Add output layer
    model.add(keras.layers.Dense(units=1, activation='sigmoid', kernel_initializer='random_normal', kernel_regularizer=kernel_regularizer))
    
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
        
    # Apply z-score to all sub-datasets
    scalable_variables = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age']
    
    # Create an instance of the StandardScaler for each variable
    scaler = preprocessing.StandardScaler()

    # Fit the distribution
    scaler.fit(x_train_valid.loc[:, scalable_variables])
    
    x_test.loc[:, scalable_variables] = scaler.transform(x_test.loc[:, scalable_variables])

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

        if scalable_variables:
            # Create an instance of the StandardScaler for each variable
            scaler = preprocessing.StandardScaler()

            # Fit the distribution
            scaler.fit(x_train.loc[:, scalable_variables])

            # Transform and normalize all variables
            x_train.loc[:, scalable_variables] = scaler.transform(x_train.loc[:, scalable_variables])
            x_valid.loc[:, scalable_variables] = scaler.transform(x_valid.loc[:, scalable_variables])
            
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
             optimizer='sgd',
             loss='binary_crossentropy',
             momentum=0,
             rho=0,
             beta_1=0,
             beta_2=0,
             learning_rate=0.1,
             decay_rate=0.1,
             batch_size=32,
             epochs=200,
             patience=50,
             min_delta=10,
             tensorboard_on=True,
             summary_on=True,
             tag='untagged',
             *args,
             **kwargs
             ):
    
    # Get current Timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create logging path
    log_dir = 'tb-logs/mlp/' + tag + '/' + timestamp
    
    # Checkpoint dir
    checkpoint_dir = 'checkpoints/mlp/' + timestamp
    
    # Create neural network
    model = create_model(*args, **kwargs)
    
    if summary_on:
        if tensorboard_on:
            print(f'Model logs @ {log_dir}')
        print(f'Model checkpoints @ {checkpoint_dir}')
        model.summary()
        
    # Create optimizer
    if optimizer == 'adam':
        model_optimizer = keras.optimizers.Adam(beta_1=beta_1, beta_2=beta_2)
    elif optimizer == 'sgd':
        model_optimizer = keras.optimizers.SGD(momentum=momentum)
    elif optimizer == 'rmsprop':
        model_optimizer = keras.optimizers.RMSprop(momentum=momentum, rho=rho)
    else:
        raise ValueError('Unknown optimizer. Try adam, sgd or rmsprop')
    
    # Compilation
    model.compile(optimizer=model_optimizer, loss=loss, metrics=['AUC'])
    
    # Create learning scheduler callback
    lr_scheduler = learningrate.ExponentialDecay(learning_rate, decay_rate)
        
    # Learning rate callback
    lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)
    
    # Model checkpoint callback
    mc_callback = keras.callbacks.ModelCheckpoint(checkpoint_dir + '.hdf5', monitor='val_auc', save_best_only=True, verbose=0, mode='max')
    
    # Tensorboard callback
    
    if tensorboard_on:
        tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')
        
    # Early stopping callback
    es_callback = keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=patience, min_delta=min_delta)
    
    # Define callbacks used
    callbacks = [es_callback, lr_callback, mc_callback]
    if tensorboard_on:
        callbacks += [tb_callback]
    
    
    # Train NN
    model.fit(x=x_train, y=y_train, validation_data=(x_valid, y_valid), epochs=epochs, verbose=0, shuffle=True, batch_size=batch_size, callbacks=callbacks, use_multiprocessing=True)
    
    # Get best result
    model = keras.models.load_model(checkpoint_dir + '.hdf5')
    
        
    # Compute metrics
    eval_train, eval_valid, eval_test = get_metrics(model, x_train, y_train, x_valid, y_valid, x_test, y_test,verbose=summary_on, f2_plot=True);
        
    return eval_train, eval_valid, eval_test