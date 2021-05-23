"""
    @file mlp_helper.py
    @author Joaquin Oscar Gaytan and Lucas Agustin Kammann
"""

# Third-party modules of Python
import tensorflow.keras as keras
import tensorflow as tf
import datetime

# Project modules of Python
import src.learningrate as learningrate


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


def create_model(layers_neurons=[], 
                 layers_activation=[],
                 use_batch_normalization=False
                ):
    """ Creates a neural network model using the given hyperparameters and the Keras framework.
        The neural network is used to solve a regression problem where the input variables are:
            * bmi
            * sex
            * region
            * children
            * age 
            * smokes
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
              batch_size=64,
              epochs=100,
              learning_rate=0.1,
              decay_rate=0.1,
              optimizer='sgd',
              patience=50,
              min_delta=10,
              summary_on=True,
              tensorboard_on=True,
              checkpoints_on=True,
              **kwargs):
    """ Creates the neural network with the given hyperparameters, compiles the model using the corresponding optimizer, loss function, 
        metrics and other hyperparameters. Train, validate and test a model built with the given settings.
        
        @param x_train, y_train          Train set
        @param x_valid, y_valid          Valid set
        @param x_test, y_test            Test set
        @param layers_neurons            List containing amount of neurons for each individual hidden layer
        @param layers_activation         List containing the activation function for each individual hidden layer
        @param use_batch_normalization   Whether batch normalization layers are added between hidden layers
        @param learning_rate             Learning rate
        @param decay_rate                Decay rate of the exponential dynamic learning rate
        @param batch_size                Batch size
        @param epochs                    Amount of epochs
        @param optimizer                 Optimizer
        @param patience                  Patience used for early stopping, amount of epochs without improvements allowed
        @param min_delta                 Minimum delta accounted as an improvement in the loss function during training
        @param tensorboard_on            Enables whether to log or not onto TensorBoard
        @param checkpoints_on            Enables whether to save model checkpoints or not
        @param summary_on                Enables whether to print a summary of the model and its results
        
        - If optimizer is 'sgd', should set the following parameters or a default value will used
        @param momentum                  Factor used with the first order momentum, belongs to [0, 1]. Default is 0.0.
        
        - If optimizer is 'rmsprop', should set the following parameters or a default value will used
        @param momentum                  Factor used with the first order momentum, belongs to [0, 1]. Default is 0.0.
        @param rho                       Factor used with the second order momentum, belongs to [0, 1]. Default is 0.0.

        - If optimizer is 'adam', should set the following parameters or a default value will be used
        @param beta_1                    Factor used with the first order momentum, belongs to [0, 1]. Default is 0.0.
        @param beta_2                    Factor used with the second order momentum, belongs to [0, 1]. Default is 0.0.
    
        @return Tuple containing model and its test performance => (model, metric)
    """
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create the logging directory name
    if tensorboard_on:
        log_dir = 'tb-logs/mlp/' + timestamp
        print(f'Model logs at {log_dir}')
    
    # Create the model checkpoint directory name
    if checkpoints_on:
        checkpoint_dir = 'checkpoints/mlp/' + timestamp
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
    
    # Create the learning rate scheduler
    if tensorboard_on:
        lr_callback = keras.callbacks.LearningRateScheduler(LRTensorBoardLogger(log_dir + '/learning-rate', learningrate.ExponentialDecay(learning_rate, decay_rate)))
    else:
        lr_callback = keras.callbacks.LearningRateScheduler(learningrate.ExponentialDecay(learning_rate, decay_rate))
        
    # Compile the neural network
    model.compile(
        optimizer=model_optimizer,
        loss=keras.losses.MAE,
        metrics=[keras.losses.MAE]
    )
    
    # Create the model checkpoint callback
    if checkpoints_on:
        mc_callback = keras.callbacks.ModelCheckpoint(checkpoint_dir + '{epoch}.hdf5', monitor='val_loss', save_best_only=True, verbose=0)
    
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
    
    # Log results
    if tensorboard_on:
        tensorboard_log(log_dir + '/testing', 'charges', y_test.to_numpy())
        tensorboard_log(log_dir + '/predicted', 'charges', model.predict(x_test).reshape(-1))
    
    # Return the trained model
    mae, _ = model.evaluate(x_test, y_test, verbose=0)
    if summary_on:
        print(f'Mean absolute error of the test set {mae}')
    return mae
