import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, Input, Dense, AvgPool2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time
from itertools import product
import pandas as pd

from utils.image_pipeline import dataset_generator
from utils.load_params import load_params
from utils.image_pipeline import datasets
from utils.custom_metrics import StatefullMultiClassFBeta
from utils.plot_metrics import plot_model_metrics

params = load_params('params.yaml')


def model_architecture(first_filter, 
                        first_kernel, 
                        second_filter,
                        second_kernel,
                        third_filter,
                        third_kernel,
                        pooling, 
                        dense_nodes, 
                        dropout, 
                        learning_rate,
                        batch_norm,
                        L2_alpha,
                        img_size):
            
    '''
    Creates dynamic three conv layer CNN model

    first_filter = first layer filter int
    first_kernel = x and y size of first kernel int (first_kernel, first_kernel)
    second_filter = second layer filter int
    second_kernel = x and y size of third kernel int (second_kernel, second_kernel)
    third_filter = third layer filter int
    third_kernel = x and y size of third kernel int (third_kernel, third_kernel)
    pooling = Max or Average pooling [layers.MaxPool2D, layers.AvgPool2D]
    dense_nodes = number of dense nodes int
    dropout = percent dropout float
    learning_rate = learning rate float
    batch_norm = batch normalization rate float
    L2_alpha = L2 regularizer float
    img_size = pixles along length and height dimension (img_size, img_size)
    '''
    
    # custom F1 score metric
    statefull_multi_class_fbeta = StatefullMultiClassFBeta()
    
    optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate, name='Nadam')
    
    # layer 1
    model = models.Sequential()
    model.add(layers.Conv2D(filters=first_filter, 
                            kernel_size=(first_kernel, first_kernel), 
                            strides=(1, 1), 
                            activation=tf.nn.relu,input_shape=(img_size, img_size, 3)))
    if batch_norm:
        model.add(layers.BatchNormalization())
        
    # Layer 2
    model.add(layers.Conv2D(filters=second_filter, 
                            kernel_size=(second_kernel, second_kernel), 
                            strides=(1, 1), 
                            activation=tf.nn.relu))
    model.add(pooling(pool_size=(2, 2),strides=2))
    model.add(layers.Dropout(dropout))
    if batch_norm:
        model.add(layers.BatchNormalization())
    
    # layer 3
    model.add(layers.Conv2D(filters=third_filter, 
                            kernel_size=(third_kernel, third_kernel), 
                            strides=(1, 1), 
                            activation=tf.nn.relu))
    model.add(pooling(pool_size=(2, 2),strides=2))
    model.add(layers.Dropout(dropout))
    model.add(layers.Flatten())
    if batch_norm:
        model.add(layers.BatchNormalization())
    
    # Dense layer
    model.add(layers.Dense(units=dense_nodes, 
                            activation=tf.nn.relu,
                           kernel_regularizer=tf.keras.regularizers.l2(L2_alpha)))
    model.add(layers.Dense(units=2, activation=tf.nn.softmax))
    
    
    model.compile(optimizer=optimizer,           
                   loss = 'categorical_crossentropy',
                   metrics=[ 'accuracy',statefull_multi_class_fbeta] #statefull_multi_class_fbeta
    )

    # print model summary
    display(model.summary())
    return model


def train_model(model, 
                   model_name, 
                   epochs, 
                   #data_import_list, 
                   img_size,
                   batch_size):
    
    train_dataset, val_dataset, test_dataset = datasets()
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, verbose=1)
    
    reduce_LR = ReduceLROnPlateau(monitor='val_accuracy', 
                                  factor=.5,
                                   patience=2, 
                                   min_lr=0.00001, 
                                   verbose=2
    )
    
    #set model artifact location if training local or in cloud
    training_locally = params.data_pipeline.training_locally

    if training_locally: 
        filepath = params.data_pipeline.pipeline_local_filepath
    else: 
        filepath = params.data_pipeline.pipeline_cloud_filepath

    save_model = ModelCheckpoint(filepath + model_name,
                                monitor="val_accuracy",
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=False,
    )

    hist = model.fit(train_dataset,
                    epochs=epochs, 
                    validation_data=val_dataset,
                    callbacks=[early_stop, reduce_LR, save_model],
                   )
    
    results = []
 
    # callbacks save the best model - need to find validation metric value in hist
    best_val_accuracy_index = np.argmax(hist.history['val_state_full_binary_fbeta'])
    for key in hist.history.keys():
        results.append(hist.history[key][best_val_accuracy_index])

    # Evaluate test dataset
    test_loss, test_acc, test_f1 = model.evaluate(test_dataset)
    results.extend([test_loss, test_acc, test_f1])
    

    plot_model_metrics(model, hist)
    
    return hist, results, model

def permute_model_parameters():
    '''
    Creates cartesian product of model parameters in params.yaml
    '''
    
    param_dict = dict(params.model_training.model_params)
    return [dict(zip(param_dict, v)) for v in product(*param_dict.values())]

def train_all_models(experiment_name, permuted_model_params, max_epochs):

    model_list = []
    model_name_list = []
    experiment_counter = int()
    model_list = [] 
    model_name_list = []

    for param in permuted_model_params:
        start_time = time.time()

        #store variables for labeling
        experiment_counter +=1
        model_name = experiment_name + " " + str(experiment_counter)
        
        first_filter = param['first_filter']
        first_kernel = param['first_kernel']
        second_filter= param['second_filter']
        second_kernel = param['second_kernel']
        third_filter= param['third_filter']
        third_kernel = param['third_kernel']
        batch_size = param['batch_size']
        pooling = param['pooling']
        optimizer = 'Nadam' 
        dense_nodes = param['dense_nodes']
        learning_rate = param['learning_rate']
        batch_norm = param['regularization'][0]
        L2_alpha = param['regularization'][1]
        dropout = param['regularization'][2]
        img_size = param['img_size']
        
        model = model_architecture(first_filter, 
                                    first_kernel, 
                                    second_filter, 
                                    second_kernel, 
                                    third_filter,
                                    third_kernel,
                                    pooling, 
                                    dense_nodes, 
                                    dropout, 
                                    learning_rate,
                                    batch_norm,
                                    L2_alpha,
                                    img_size
        )
            
        hist, results, fit_model = train_model(model, 
                                           model_name, 
                                           max_epochs, 
                                           img_size,
                                           batch_size
        )

        # legnth of any hist value = number of epochs                           
        epoch_count = len(hist.history['loss'])
        
        # round all results to precision of 3
        results = [round(result,3) for result in results]

        end_time = time.time()
        experiment_duration = end_time - start_time

        # create results record for this training run
        param_list = []
        param_list.extend([model_name, 
                           first_filter, 
                           first_kernel, 
                           second_filter,
                           second_kernel,
                           third_filter,
                           third_kernel,
                           dense_nodes,
                           dropout,
                           str(batch_norm),
                           L2_alpha,
                           learning_rate,
                           epoch_count, 
                           experiment_duration,
                           optimizer, 
                           batch_size,
                           model.count_params()
        ])
        metrics_param_list =  param_list + results
        
        # the first run must create df and define columns, all other runs can append to df
        if experiment_counter == 1:
            results_df = pd.DataFrame([metrics_param_list], columns = ['Experiment Name',
                                                                    'First Filter', 
                                                                    'First Kernel',
                                                                    'Second Filter',
                                                                    'Second Kernel',
                                                                    'Third Filter',
                                                                    'Third Kernel',
                                                                    'Dense Nodes',
                                                                    'Dropout Rate',
                                                                    'Batch Norm T/F',
                                                                    'L2 alpha',
                                                                    'Learning Rate',
                                                                    'Completed Epochs',
                                                                    'Duration (seconds)',
                                                                    'Optimizer',
                                                                    'Batch Size',
                                                                    'Model Parameters',
                                                                    'loss', 
                                                                    'acc', 
                                                                    'F1', 
                                                                    'val_loss', 
                                                                    'val_acc',
                                                                    'val_F1', 
                                                                    'test_loss', 
                                                                    'test_acc', 
                                                                    'test_F1',
            ])
        else:
            df_length = len(results_df)
            results_df.loc[df_length] = metrics_param_list
        
        print(model_name, " Runs left, ", len(permuted_model_params)-experiment_counter)
        
        # store fit model object for later analysis
        model_list.append(fit_model)
        model_name_list.append(model_name)

    return [results_df, model_list, model_name_list]

