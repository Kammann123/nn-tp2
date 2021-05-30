import tensorflow.keras as keras
import tensorflow as tf
import datetime

import src.learningrate as learningrate
import src.helper as helper

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
             *args,
             **kwargs
             ):
    
    # Get current Timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create logging path
    log_dir = 'tb-logs/mlp/' + timestamp
    
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
    
    if tensorboard_on:
        print('Tensorboard not supported so far... Add Callback!')
        
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
    
    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)
    print(x_test.shape, y_test.shape)
    
    # Train NN
    model.fit(x=x_train, y=y_train, validation_data=(x_valid, y_valid), epochs=epochs, verbose=0, shuffle=True, batch_size=batch_size, callbacks=callbacks, use_multiprocessing=True)
    
    # Get best result
    model = keras.models.load_model(checkpoint_dir + '.hdf5')
    
    # Log results
    if tensorboard_on:
        print('Logging should be here... Add Callback!')
    
    # Compute metrics
    eval_train = model.evaluate(x=x_train, y=y_train, return_dict=True)
    eval_valid = model.evaluate(x=x_valid, y=y_valid, return_dict=True)    
    eval_test = model.evaluate(x=x_test, y=y_test, return_dict=True)
    
    auc_train = eval_train['auc']
    auc_valid = eval_valid['auc']
    auc_test = eval_test['auc']
    
    if summary_on:
        print(f'[AUC] Train = {auc_train:.4f} - Valid = {auc_valid:.4f} - Test = {auc_test:.4f}')
        
    return auc_train, auc_valid, auc_test