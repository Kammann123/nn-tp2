"""
    @file   mlp_helper.py
    @desc   Contains specific functions and classes to be used in the MLP problem
    @author Joaquin Oscar Gaytan and Lucas Agustin Kammann
"""

# Third-party modules of Python
from sklearn import model_selection
from sklearn import preprocessing
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import datetime

# Project modules of Python
import src.learningrate as learningrate
import src.helper as helper


def create_model(hidden_layers=0,
                 units_per_layer=0,
                 hidden_layer_activation=None,
                 regularizer=None,
                 regularizer_lambda=1e-4,
                 bias_initializer='zeros',
                 kernel_initializer='random_normal',
                 dropout_rate=0.0,
                 use_batch_normalization=False
                ):
    """ Creates a neural network model.
        The neural network is used to solve a regression problem using five numerical variables 
        and one categorical variable which will be encoded using an embedding layer.
        
        @param hidden_layers             Number of hidden layers
        @param units_per_layer           Units or neurons per layer
        @param hidden_layer_activation   Activation function used in hidden layers
        @param regularizer               Type of regularizer to use, values supported are None, 'l1' or 'l2'
        @param regularizer_lambda        Regularizer coefficient
        @param bias_initializer          Initializer for bias weights
        @param kernel_initializer        Initializer of synaptic weights
        @param dropout_rate              Rate of the dropout layer added after each dense layer
        @param use_batch_normalization   Determines whether to use Batch Normalization between hidden layers or not
        @return Keras neural network or model instance
    """
    # If a regularizer is set, create it before creating the model
    if regularizer == 'l1':
        kernel_regularizer = keras.regularizers.l1(regularizer_lambda)
    elif regularizer == 'l2':
        kernel_regularizer = keras.regularizers.l2(regularizer_lambda)
    else:
        kernel_regularizer = None

    # Create two input layers for the categorical and the numerical variables
    x1 = keras.layers.Input(shape=(5, ))
    x2 = keras.layers.Input(shape=(1, ))

    # Create an embedding layer with the categorical variable and create
    # a new input layer for the neural network, combining both the embedding and the 
    # numerical variable input layer
    embedding = keras.layers.Embedding(4, 2, input_length=1, embeddings_initializer='normal')(x2)
    flatten = keras.layers.Flatten()(embedding)
    x = keras.layers.Concatenate()([x1, flatten])
    current_layer = x

    # Add the hidden layers to the neural network
    layers = []
    for layer_index in range(hidden_layers):
        # Create a dense layer and add it to the neural network
        layers_per_layer = 1 + (1 if dropout_rate else 0) + (1 if use_batch_normalization else 0)
        previous_layer = layers[layers_per_layer * layer_index - 1] if layer_index else x
        current_layer = keras.layers.Dense(units=units_per_layer, 
                                           activation=hidden_layer_activation, 
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           kernel_regularizer=kernel_regularizer
                                          )(previous_layer)
        layers.append(current_layer)
        
        # Create a dropout layer aftear each dense layer
        if dropout_rate:
            previous_layer = current_layer
            current_layer = keras.layers.Dropout(dropout_rate)(previous_layer)
            layers.append(current_layer)
        
        # Create a batch normalization layer after each dense layer
        if use_batch_normalization:
            previous_layer = current_layer
            current_layer = keras.layers.BatchNormalization()(previous_layer)
            layers.append(current_layer)

    # Add the output layer
    y = keras.layers.Dense(units=1,
                           activation='linear',
                           kernel_initializer='random_normal',
                           kernel_regularizer=kernel_regularizer
                          )(current_layer)

    # Create the neural network model
    return keras.Model(inputs=[x1, x2], outputs=y)


def run_model(x_train, y_train, x_valid, y_valid, x_test, y_test,
              optimizer='sgd',
              loss='mae',
              momentum=0,
              rho=0,
              beta_1=0,
              beta_2=0,
              learning_rate=0.1,
              decay_rate=0.1,
              batch_size=64,
              epochs=100,
              patience=50,
              min_delta=10,
              tensorboard_on=True,
              summary_on=True,
              verbose=0,
              tag='experiment',
              *args,
              **kwargs
             ):
    """ Creates the neural network with the given hyperparameters, compiles the model using the corresponding optimizer, loss function, 
        metrics and other hyperparameters. Train, validate and test a model built with the given settings.
        
        @param x_train, y_train          Train set
        @param x_valid, y_valid          Valid set
        @param x_test, y_test            Test set
        @param optimizer                 Optimizer
        @param loss                      Loss function to be used
        @param momentum                  Factor used with the first order momentum, belongs to [0, 1]. Default is 0
        @param rho                       Factor used with the second order momentum, belongs to [0, 1]. Default is 0
        @param beta_1                    Factor used with the first order momentum, belongs to [0, 1]. Default is 0
        @param beta_2                    Factor used with the second order momentum, belongs to [0, 1]. Default is 0
        @param learning_rate             Learning rate
        @param decay_rate                Decay rate of the exponential dynamic learning rate
        @param batch_size                Batch size
        @param epochs                    Amount of epochs
        @param patience                  Patience used for early stopping, amount of epochs without improvements allowed
        @param min_delta                 Minimum delta accounted as an improvement in the loss function during training
        @param tensorboard_on            Enables whether to log or not onto TensorBoard
        @param summary_on                Enables whether to print a summary of the model and its results
        @param verbose                   Passes the verbose to the .fit() routine from the Keras framework
        @param tag                       Tag name used to identify in the logs and checkpoints
        @param *args, **kwargs           Parameters passed to the model
        @return MAE measured in train, valid and test (mae_train, mae_valid, mae_test)
    """
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create the logging directory name
    log_dir = f'tb-logs/mlp/{tag}/{timestamp}'
    
    # Create the model checkpoint directory name
    checkpoint_dir = f'checkpoints/mlp/{tag}/{timestamp}'
    
    # Create the neural network
    model = create_model(*args, **kwargs)
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
        
    # Compile the neural network
    model.compile(optimizer=model_optimizer,
                  loss=loss,
                  metrics=[ keras.losses.MAE ]
                 )
    
    # Create the learning rate scheduler and callback
    lr_scheduler = learningrate.ExponentialDecay(learning_rate, decay_rate)
    
    if tensorboard_on:
        lr_scheduler = helper.LRTensorBoardLogger(log_dir + '/learning-rate', lr_scheduler)
    
    lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)
    
    # Create the model checkpoint callback
    mc_callback = keras.callbacks.ModelCheckpoint(checkpoint_dir + '.hdf5',
                                                  monitor='val_loss',
                                                  save_best_only=True,
                                                  verbose=0,
                                                  mode='min'
                                                 )
    
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
    callbacks = [ es_callback, lr_callback, mc_callback ]
    if tensorboard_on:
        callbacks += [ tb_callback ]
    
    # Train the neural network
    model.fit(x_train, y_train,
              validation_data=(x_valid, y_valid),
              epochs=epochs, 
              verbose=verbose, 
              shuffle=True, 
              batch_size=batch_size,
              callbacks=callbacks,
              use_multiprocessing=True
             )
    
    # Load the best model and evaluate the metric
    model = keras.models.load_model(checkpoint_dir + '.hdf5')
    
    # Log results
    if tensorboard_on:
        helper.tensorboard_log(log_dir + '/train/true', 'charges', y_train.to_numpy())
        helper.tensorboard_log(log_dir + '/train/predicted', 'charges', model.predict(x_train).reshape(-1))
        helper.tensorboard_log(log_dir + '/test/true', 'charges', y_test.to_numpy())
        helper.tensorboard_log(log_dir + '/test/predicted', 'charges', model.predict(x_test).reshape(-1))
        
    # Copute the test set metric
    mae_train, _ = model.evaluate(x_train, y_train, verbose=0)
    mae_valid, _ = model.evaluate(x_valid, y_valid, verbose=0)
    mae_test, _ = model.evaluate(x_test, y_test, verbose=0)
    
    # Round values
    mae_train = round(mae_train, 2)
    mae_valid = round(mae_valid, 2)
    mae_test = round(mae_test, 2)
    
    if summary_on:
        print(f'[MAE] Train: {mae_train} Valid: {mae_valid} Test: {mae_test}')
    return mae_train, mae_valid, mae_test


def run_model_with_kfold(x, y, test_size, n_splits, random_state=10, *args, **kwargs):
    """ Run model using K-Fold validation method to improve metric estimation.
        @param x, y               Dataset
        @param test_size          Relative size of the dataset to be used for test
        @param n_splits           Number of slipts for the k-fold validation method
        @param *args, **kargs     Parameters used for the model
        @return MAE measured in train, valid and test (mae_train, mae_valid, mae_test)
    """
    
    # Split the dataset into train_valid and test
    x_train_valid, x_test_un, y_train_valid, y_test_un = model_selection.train_test_split(x, y, test_size=test_size, random_state=random_state, shuffle=True)
    
    # Apply the z-score to the test set and re-arange for a suitable order of variables
    # scalable_variables = ['bmi', 'age']
    # if scalable_variables:
    #    scaler = preprocessing.StandardScaler()
    #    scaler.fit(x_train_valid.loc[:, scalable_variables])
    #    x_test.loc[:, scalable_variables] = scaler.transform(x_test.loc[:, scalable_variables])
    #x_test = [x_test[['age', 'bmi', 'smoker-encoded', 'children', 'sex-encoded']], x_test['region-encoded']]
    
    # Create an instance of a K-Folding handler
    kf = model_selection.KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    # Generate arrays to contain train, valid and test metrics
    train_metrics = np.zeros(n_splits)
    valid_metrics = np.zeros(n_splits)
    test_metrics = np.zeros(n_splits)

    # Iterate through each fold
    for i, (train, valid) in enumerate(kf.split(x_train_valid, y_train_valid)):

        # Create a copy of the train and valid sets used for the current fold or iteration
        x_train = x_train_valid.iloc[train].copy()
        y_train = y_train_valid.iloc[train].copy()
        x_valid = x_train_valid.iloc[valid].copy()
        y_valid = y_train_valid.iloc[valid].copy()
        x_test = x_test_un.copy()
        y_test = y_test_un.copy()

        # Apply the z-score to normalize both the train and valid sets, and re-arange features
        scalable_variables = ['bmi', 'age']
        if scalable_variables:
            scaler = preprocessing.StandardScaler()
            scaler.fit(x_train.loc[:, scalable_variables])
            x_train.loc[:, scalable_variables] = scaler.transform(x_train.loc[:, scalable_variables])
            x_valid.loc[:, scalable_variables] = scaler.transform(x_valid.loc[:, scalable_variables])
            x_test.loc[:, scalable_variables] = scaler.transform(x_test.loc[:, scalable_variables])
        x_train = [x_train[['age', 'bmi', 'smoker-encoded', 'children', 'sex-encoded']], x_train['region-encoded']]
        x_valid = [x_valid[['age', 'bmi', 'smoker-encoded', 'children', 'sex-encoded']], x_valid['region-encoded']]
        x_test = [x_test[['age', 'bmi', 'smoker-encoded', 'children', 'sex-encoded']], x_test['region-encoded']]

        # Run model and save metrics
        mae_train, mae_valid, mae_test = run_model(x_train, y_train, x_valid, y_valid, x_test, y_test, *args, **kwargs)
        train_metrics[i] = mae_train
        valid_metrics[i] = mae_valid
        test_metrics[i] = mae_test
    
    # Return results
    return train_metrics, valid_metrics, test_metrics