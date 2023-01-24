from itertools import product
from utils.data_pipeline_utils import load_params
from tensorflow.keras import layers
from types import SimpleNamespace    
from tensorflow.keras.models import Sequential
from tensorflow.keras import models
import tensorflow as tf



def model_architecture(model_param_dict):

    # creates variable using the key and assigns the value to that new variable 
    #for key, value in model_param_dict.items():
    #   exec(f'{key}={value}', globals())
    #print(globals().keys())
    #print(first_filter.__name__)

    return SimpleNamespace(**model_param_dict)
    

class ModelTraining:

    def __init__(self) -> None:
        self.params = load_params()
    
    def permute_model_parameters(self):
        '''
        Creates cartesian product of model parameters in params.yaml
        '''
        
        param_dict = dict(self.params.model_training.model_params)
        permuted_params = [dict(zip(param_dict, v)) for v in product(*param_dict.values())]

        return permuted_params[:3]
    
    def CNN_layer(self, model:models.Sequential,
                first_layer_bool:bool,
                filter_size:int,
                kernel_size:int,
                pooling:layers,
                dropout:float,
                batch_norm:bool) -> models.Sequential:
        
        if first_layer_bool:
            model.add(layers.Conv2D(filter_size=filter_size,
                                    kernel_size=(kernel_size, kernel_size),
                                    strides=(1, 1),
                                    activation=tf.nn.relu,
                                    input_shape=(img_size, img_size, 3)))
        else: 
            model.add(layers.Conv2D(filter_size=filter_size,
                                kernel_size=(kernel_size, kernel_size),
                                strides=(1, 1),
                                activation=tf.nn.relu))

        model.add(pooling(pool_size=(2, 2),strides=2))
        model.add(layers.Dropout(dropout))
        if batch_norm:
            model.add(layers.BatchNormalization())
        return model

    def model_architecture(self, model_param_dict):
        # dynamically creates variables in the scope of this function
        # using the dict key and assigns the value to that new variable 
        #for key, value in model_param_dict.items():
           # exec(f'{key}={value}', globals())

        # regularization variable is generated from above exec statement

        

        first_layer = model_param_dict['first_filter']
        

        print(first_layer)
        
