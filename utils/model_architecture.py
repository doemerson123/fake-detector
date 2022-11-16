from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, Input, Dense, AvgPool2D

from utils.custom_metrics import StatefullMultiClassFBeta

def CNN_model(first_filter, 
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
    display(model.summary())
    return model