"""
    @file   mlp_helper.py
    @desc   Contains specific functions and classes to be used in the MLP problem
    @author Joaquin Oscar Gaytan and Lucas Agustin Kammann
"""

# Third-party modules of Python
import tensorflow.keras as keras
import tensorflow as tf
import datetime

# Project modules of Python
import src.learningrate as learningrate
import src.helper as helper


def create_model(layers_neurons=[], 
                 layers_activation=[],
                 use_batch_normalization=False
                ):
    """ Creates a neural network model using the given hyperparameters and the Keras framework.
        The neural network is used to solve a regression problem where the input variables are:
            * bmi
            * sex
            * children
            * age 
            * smokes
            * region
        And the output variable is the amount of money charged for medical expenses.
        
        @param layers_neurons            List containing amount of neurons for each individual hidden layer
        @param layers_activation         List containing the activation function for each individual hidden layer
        @param use_batch_normalization   Determines whether to use Batch Normalization between hidden layers or not
        
        @return Keras neural network or model instance
    """

    # Validate the amount of layers described
    if len(layers_neurons) != len(layers_activation):
        raise ValueError('Invalid amount of layers in both the neurons and the activation function description')

    # Internal class parameters
    layer_count = len(layers_neurons)

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
    for layer_index in range(layer_count):
        # Create a dense layer and add it to the neural network
        layer_neurons = layers_neurons[layer_index]
        layer_activation = layers_activation[layer_index]
        previous_layer = layers[(2 * layer_index - 1) if use_batch_normalization else (layer_index - 1)] if layer_index else x
        current_layer = keras.layers.Dense(units=layer_neurons, 
                                           activation=layer_activation, 
                                           kernel_initializer='random_normal', 
                                           bias_initializer='zeros'
                                          )(previous_layer)
        layers.append(current_layer)
        
        # Create a batch normalization layer at the output of the previous dense layer
        if use_batch_normalization:
            previous_layer = current_layer
            current_layer = keras.layers.BatchNormalization()(previous_layer)
            layers.append(current_layer)

    # Add the output layer
    y = keras.layers.Dense(units=1, activation='linear')(current_layer)

    # Create the neural network model
    return keras.Model(inputs=[x1, x2], outputs=y)


def run_model(x_train, y_train, x_valid, y_valid, x_test, y_test, 
              layers_neurons=[], 
              layers_activation=[],
              use_batch_normalization=False,
              optimizer='sgd',
              learning_rate=0.1,
              batch_size=64,
              epochs=100,
              decay_rate=0.1,
              patience=50,
              min_delta=10,
              tensorboard_on=True,
              summary_on=True,
              **kwargs):
    """ Creates the neural network with the given hyperparameters, compiles the model using the corresponding optimizer, loss function, 
        metrics and other hyperparameters. Train, validate and test a model built with the given settings.
        
        @param x_train, y_train          Train set
        @param x_valid, y_valid          Valid set
        @param x_test, y_test            Test set
        @param layers_neurons            List containing amount of neurons for each individual hidden layer
        @param layers_activation         List containing the activation function for each individual hidden layer
        @param use_batch_normalization   Whether batch normalization layers are added between hidden layers
        @param optimizer                 Optimizer
        @param learning_rate             Learning rate
        @param decay_rate                Decay rate of the exponential dynamic learning rate
        @param batch_size                Batch size
        @param epochs                    Amount of epochs
        @param patience                  Patience used for early stopping, amount of epochs without improvements allowed
        @param min_delta                 Minimum delta accounted as an improvement in the loss function during training
        @param tensorboard_on            Enables whether to log or not onto TensorBoard
        @param summary_on                Enables whether to print a summary of the model and its results
        
        [ optimizer options ]
        - If optimizer is 'sgd', should set the following parameters or a default value will be used
        @param momentum                  Factor used with the first order momentum, belongs to [0, 1]. Default is 0
        
        - If optimizer is 'rmsprop', should set the following parameters or a default value will be used
        @param momentum                  Factor used with the first order momentum, belongs to [0, 1]. Default is 0
        @param rho                       Factor used with the second order momentum, belongs to [0, 1]. Default is 0

        - If optimizer is 'adam', should set the following parameters or a default value will be used
        @param beta_1                    Factor used with the first order momentum, belongs to [0, 1]. Default is 0
        @param beta_2                    Factor used with the second order momentum, belongs to [0, 1]. Default is 0
        
        @return Mean absolute error in the given test set
    """
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create the logging directory name
    if tensorboard_on:
        log_dir = 'tb-logs/mlp/' + timestamp
        if summary_on:
            print(f'Model logs at {log_dir}')
    
    # Create the model checkpoint directory name
    checkpoint_dir = 'checkpoints/mlp/' + timestamp
    if summary_on:
        print(f'Model checkpoints at {checkpoint_dir}')
    
    # Create the neural network
    model = create_model(layers_neurons, layers_activation, use_batch_normalization)
    if summary_on:
        model.summary()
    
    # Create the optimizer
    if optimizer == 'adam':
        beta_1 = kwargs['beta_1'] if 'beta_1' in kwargs.keys() else 0.0
        beta_2 = kwargs['beta_2'] if 'beta_2' in kwargs.keys() else 0.0
        model_optimizer = keras.optimizers.Adam(beta_1=beta_1, beta_2=beta_2)
    elif optimizer == 'sgd':
        momentum = kwargs['momentum'] if 'momentum' in kwargs.keys() else 0.0
        model_optimizer = keras.optimizers.SGD(momentum=momentum)
    elif optimizer == 'rmsprop':
        momentum = kwargs['momentum'] if 'momentum' in kwargs.keys() else 0.0
        rho = kwargs['rho'] if 'rho' in kwargs.keys() else 0.0
        model_optimizer = keras.optimizers.RMSprop(momentum=momentum, rho=rho)
    else:
        raise ValueError('Unknown or unsupported optimizer, expected: adam, sgd or rmsprop')
        
    # Compile the neural network
    model.compile(optimizer=model_optimizer,
                  loss=keras.losses.MAE,
                  metrics=[ keras.losses.MAE ]
                 )
    
    # Create the learning rate scheduler and callback
    lr_scheduler = learningrate.ExponentialDecay(learning_rate, decay_rate)
    
    if tensorboard_on:
        lr_scheduler = helper.LRTensorBoardLogger(log_dir + '/learning-rate', lr_scheduler)
    
    lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)
    
    # Create the model checkpoint callback
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
    callbacks = [ es_callback, lr_callback, mc_callback ]
    if tensorboard_on:
        callbacks += [ tb_callback ]
    
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
        helper.tensorboard_log(log_dir + '/testing', 'charges', y_test.to_numpy())
        helper.tensorboard_log(log_dir + '/predicted', 'charges', model.predict(x_test).reshape(-1))
        
    # Copute the test set metric
    mae, _ = model.evaluate(x_test, y_test, verbose=0)
    if summary_on:
        print(f'Mean absolute error of the test set {mae}')
    return mae
