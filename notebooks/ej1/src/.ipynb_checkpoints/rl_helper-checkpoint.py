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
    print(f'Degree = {degree}')
    sh = x_train.shape
    print(f'x_train shape = {sh}')
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
        
    #if tensorboard_on:
        #lr_scheduler = helper.LRTensorBoardLogger(log_dir + '/learning-rate', lr_scheduler)
        #print('uncomment line 123!')
    
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
    
    # Log results
    #if tensorboard_on:
        #helper.tensorboard_log(log_dir + '/train/true', 'charges', y_train.to_numpy().reshape(-1))
        #helper.tensorboard_log(log_dir + '/train/predicted', 'charges', model.predict(x_train).reshape(-1))
        #helper.tensorboard_log(log_dir + '/test/true', 'charges', y_test.to_numpy())
        #helper.tensorboard_log(log_dir + '/test/predicted', 'charges', model.predict(x_test).reshape(-1))
        #print('uncomment line 166!')
        
    # Compute metrics
    eval_train = model.evaluate(x_train, y_train, verbose=0, return_dict=True)
    eval_valid = model.evaluate(x_valid, y_valid, verbose=0, return_dict=True)
    eval_test = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    
    train_scores = {'auc': 0}
    valid_scores = {'auc': 0}    
    test_scores = {'auc': 0}
    

    auc_train = eval_train['auc']
    auc_valid= eval_valid['auc']
    auc_test = eval_test['auc']
    
    
    # Assign metrics
    train_scores['auc'] = auc_train
    valid_scores['auc'] = auc_valid
    test_scores['auc'] = auc_test
    
    if summary_on:
        print(f'[AUC] Train: {auc_train:.4f} Valid: {auc_valid:.4f} Test: {auc_test:.4f}')
        
    return auc_train, auc_valid, auc_test
    