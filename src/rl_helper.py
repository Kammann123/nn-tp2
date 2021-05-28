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

# Project modules of Python
import src.learningrate as learningrate
import src.helper as helper


def create_model(input_shape=1,
                 regularizer=None,
                 regularizer_lambda=1e-10):
    """ Creates a linear regression model using the given hyperparameters.
    
        @param input_shape               Amount of input variables of the model
        @param regularizer               Type of regularizer to use, values supported are None, 'l1' or 'l2'
        @param regularizer_lambda        Regularizer coefficiente
        @return Keras model instance
    """
    # If a regularizer is set, create it before creating the model
    if regularizer == 'l1':
        kernel_regularizer = keras.regularizers.l1(regularizer_lambda)
    elif regularizer == 'l2':
        kernel_regularizer = keras.regularizers.l2(regularizer_lambda)
    else:
        kernel_regularizer = None
        
    # Create the linear regression model
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_shape, )))
    model.add(keras.layers.Dense(units=1, 
                                 activation='linear', 
                                 kernel_initializer='random_normal',
                                 kernel_regularizer=kernel_regularizer
                                ))
    return model


def run_model(x_train, y_train, x_valid, y_valid, x_test, y_test,
              learning_rate,
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
              batch_size=64,
              epochs=100,
              patience=50,
              min_delta=10,
              tensorboard_on=True,
              checkpoints_on=True,
              summary_on=True,
              *args,
              **kwargs):
    """ Creates the model using the given hyperparameters, compiles the model and run the train, 
        validation and test process.
        
        @param x_train, y_train          Train set
        @param x_valid, y_valid          Valid set
        @param x_test, y_test            Test set
        @param learning_rate             Learning rate
        @param degree                    Order of the polynomial features
        @param scheduler                 Type of learning rate used, can be None (constant learning rate), 'exponential-decay', 'step-decay' or 'time-decay'
        @param decay_rate                Decaying rate of the learning rate, when using TimeBasedDecay or ExponentialDecay
        @param drop_rate                 Drop rate of the learning rate, when using StepDecay
        @param epochs_drop               Period of epochs for learning rate update, when using StepDecay
        @param optimizer                 Optimizer
        @param momentum                  Factor used with the first order momentum, belongs to [0, 1]. Default is 0
        @param rho                       Factor used with the second order momentum, belongs to [0, 1]. Default is 0
        @param beta_1                    Factor used with the first order momentum, belongs to [0, 1]. Default is 0
        @param beta_2                    Factor used with the second order momentum, belongs to [0, 1]. Default is 0
        @param batch_size                Batch size
        @param epochs                    Amount of epochs
        @param patience                  Patience used for early stopping, amount of epochs without improvements allowed
        @param min_delta                 Minimum delta accounted as an improvement in the loss function during training
        @param tensorboard_on            Enables whether to log or not onto TensorBoard
        @param checkpoints_on            Enables whether to save model checkpoints or not
        @param summary_on                Enables whether to print a summary of the model and its results
        @return Mean absolute error in the given test set
    """
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create the logging directory name
    log_dir = 'tb-logs/rl/' + timestamp
    
    # Create the model checkpoint directory name
    checkpoint_dir = 'checkpoints/rl/' + timestamp
    
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
                  loss=keras.losses.MAE,
                  metrics=[ keras.losses.MAE ]
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
    
    if tensorboard_on:
        lr_scheduler = helper.LRTensorBoardLogger(log_dir + '/learning-rate', lr_scheduler)
    
    lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)
    
    # Create the model checkpoint callback
    if checkpoints_on:
        mc_callback = keras.callbacks.ModelCheckpoint(checkpoint_dir + '.hdf5', monitor='val_loss', save_best_only=True, verbose=0)
    
    # Create the tensorboard callback
    if tensorboard_on:
        tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                  histogram_freq=1,
                                                  embeddings_freq=1, 
                                                  update_freq='epoch'
                                                 )
    
    # Create the early stopping callback
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                mode='min',
                                                patience=patience,
                                                min_delta=min_delta
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
    if tensorboard_on:
        helper.tensorboard_log(log_dir + '/train/true', 'charges', y_train.to_numpy().reshape(-1))
        helper.tensorboard_log(log_dir + '/train/predicted', 'charges', model.predict(x_train).reshape(-1))
        helper.tensorboard_log(log_dir + '/test/true', 'charges', y_test.to_numpy())
        helper.tensorboard_log(log_dir + '/test/predicted', 'charges', model.predict(x_test).reshape(-1))
        
    # Copute the test set metric
    mae_train, _ = model.evaluate(x_train, y_train, verbose=0)
    mae_valid, _ = model.evaluate(x_valid, y_valid, verbose=0)
    mae_test, _ = model.evaluate(x_test, y_test, verbose=0)
    if summary_on:
        print(f'[MAE] Train: {mae_train} Valid: {mae_valid} Test: {mae_test}')
    return mae_train, mae_valid, mae_test
