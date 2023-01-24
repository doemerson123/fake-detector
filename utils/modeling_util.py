import time
from itertools import product
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras import models, layers
from tensorflow.keras.layers import (Conv2D, MaxPool2D, BatchNormalization, 
                                    Dropout, Flatten, Input, Dense, AvgPool2D)
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, 
                                        ReduceLROnPlateau)
from utils.custom_metrics_util import StatefullMultiClassFBeta
from utils.plot_metrics_util import plot_model_metrics
from utils.data_pipeline_utils import load_params
from utils.data_pipeline_util import create_dataset, filepath





params = load_params()
img_size = params.model_training.global_params.img_size
permuted_training_parameters = permute_model_parameters()

class ModelTraining:

    def __init__(self) -> None:
        self.params = load_params()
        self.img_size = self.params.model_training.global_params.img_size

    def permute_model_parameters(self, convolution_layers):
        '''
        Creates cartesian product of model parameters from params.yaml

        Requires convolution_layers to be valued at 2 or 3
        '''
        
        param_dict = dict(self.params.model_training.model_params)
        
        # remove third layer parameters if not needed
        if convolution_layers == 2:
            del param_dict['third_filter']
            del param_dict['third_kernel']

        return [dict(zip(param_dict, v)) for v in \
                        product(*param_dict.values())]

    def conv_layer(self,
                model:models.Sequential,
                params:dict,
                filter_size:int,
                kernel_size:int,
                first_layer_bool:bool) -> models.Sequential:
        
        if first_layer_bool:
            model.add(layers.Conv2D(filter_size=filter_size,
                                    kernel_size=(kernel_size, kernel_size),
                                    strides=(1, 1),
                                    activation=tf.nn.relu,
                                    input_shape=(self.img_size, 
                                                self.img_size, 3)))
        else: 
            model.add(layers.Conv2D(filter_size=filter_size,
                                kernel_size=(kernel_size, kernel_size),
                                strides=(1, 1),
                                activation=tf.nn.relu))

        model.add(params['pooling'](pool_size=(2, 2),strides=2))
        model.add(layers.Dropout(params['dropout']))
        if params['batch_norm']:
            model.add(layers.BatchNormalization())
        return model

    def dense_layer(self,
                    model:models.Sequential,
                    params:dict) -> models.Sequential:
        
        batch_norm = params['regularization'][0]
        
        if batch_norm:
            model.add(layers.BatchNormalization())

        model.add(layers.Dense(units=params['dense_nodes'],
                        activation=tf.nn.relu,
                        kernel_regularizer= \
                            tf.keras.regularizers.l2(params['l2_alpha'])))
        return model

    def model_architecture(self, params, convolution_layers, dense_layers):


        # custom F1 metric
        statefull_multi_class_fbeta = StatefullMultiClassFBeta()

        optimizer = tf.keras.optimizers.Nadam(learning_rate = \
                                                params['learning_rate'], 
                                            name='Nadam')
        
        model = models.Sequential()

        # CNN layers
        if convolution_layers == 2:
            conv_layer_order_list = ['first', 'second']
        if convolution_layers == 3:
            conv_layer_order_list = ['first', 'second', 'third']

        for conv_layer in conv_layer_order_list:
            first_layer_bool = False
            if 'first' in conv_layer:
                first_layer_bool = True
                
            model = self.conv_layer(model,
                            params,
                            params[conv_layer + '_filter'],
                            params[conv_layer + '_kernel'],
                            first_layer_bool)
        
        # dense layers
        model.add(layers.Flatten())
        for dense_layers in range(dense_layers):
            model = self.dense_layer(model, params)
        model.add(layers.Dense(units=2, activation=tf.nn.softmax))
        
        # compile
        model.compile(optimizer=optimizer,           
                    loss = 'categorical_crossentropy',
                    metrics=['accuracy', statefull_multi_class_fbeta])
        return model

    def train_model(self, model, model_name):
        
        train_dataset = create_dataset('train')
        val_dataset = create_dataset('val')
        test_dataset = create_dataset('test')
        
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                    patience=4,
                                                    verbose=1)
        reduce_LR = ReduceLROnPlateau(monitor='val_accuracy',
                                    factor=.5,
                                    patience=2, 
                                    min_lr=0.00001, 
                                    verbose=2)
        
        # set model artifact location 
        checkpoint_filepath, _ = filepath('checkpoint')

        checkpoint = ModelCheckpoint(checkpoint_filepath + model_name,
                                    monitor="val_accuracy",
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=False)

        hist = model.fit(train_dataset,
                        epochs=epochs, 
                        validation_data=val_dataset,
                        callbacks=[early_stop, reduce_LR, checkpoint])
        
        results = []
    
        # callbacks save best model - retreive best validation metric from hist
        best_val_accuracy_index = np.argmax(hist.history['val_state_full_binary_fbeta'])
        for key in hist.history.keys():
            results.append(hist.history[key][best_val_accuracy_index])

        # Evaluate test dataset and plot
        test_loss, test_acc, test_f1 = model.evaluate(test_dataset)
        results.extend([test_loss, test_acc, test_f1])
        plot_model_metrics(model_name, model, hist)
        
        return hist, results, model


    def train_all_models(experiment_name, permuted_model_params):

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
                                        img_size)
                
            hist, results, fit_model = train_model(model, model_name)

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
                            model.count_params()])
            metrics_param_list =  param_list + results
            
            # first run creates df and names columns, each run after extends df
            if experiment_counter == 1:
                results_df = pd.DataFrame([metrics_param_list], 
                                        columns = ['Experiment Name',
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
                                                    'test_F1'])
            else:
                df_length = len(results_df)
                results_df.loc[df_length] = metrics_param_list
            
            # store fit model object for later analysis
            model_list.append(fit_model)
            model_name_list.append(model_name)

        return [results_df, model_list, model_name_list]

