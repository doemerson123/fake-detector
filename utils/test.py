from itertools import product
from utils.data_pipeline_utils import load_params
from tensorflow.keras import layers
from types import SimpleNamespace
from tensorflow.keras.models import Sequential
from tensorflow.keras import models
import tensorflow as tf


def model_architecture(model_param_dict):

    # creates variable using the key and assigns the value to that new variable
    # for key, value in model_param_dict.items():
    #   exec(f'{key}={value}', globals())
    # print(globals().keys())
    # print(first_filter.__name__)

    return SimpleNamespace(**model_param_dict)


class ModelTraining:
    def __init__(self):
        self.params = load_params()

    #    def permute_model_parameters(self):
    #        '''
    #        Creates cartesian product of model parameters in params.yaml
    #        '''
    #
    #        param_dict = dict(self.params.model_training.model_params)
    #        permuted_params = [dict(zip(param_dict, v)) for v in product(*param_dict.values())]#

    # return permuted_params[:3]

    def CNN_layer(
        self,
        model: Sequential,
        first_layer_bool: bool,
        filter_size: int,
        kernel_size: int,
        pooling: layers,
        dropout: float,
        batch_norm: bool,
    ) -> Sequential:

        if first_layer_bool:
            model.add(
                Conv2D(
                    filter_size=filter_size,
                    kernel_size=(kernel_size, kernel_size),
                    strides=(1, 1),
                    activation=tf.nn.relu,
                    input_shape=(img_size, img_size, 3),
                )
            )
        else:
            model.add(
                Conv2D(
                    filter_size=filter_size,
                    kernel_size=(kernel_size, kernel_size),
                    strides=(1, 1),
                    activation=tf.nn.relu,
                )
            )

        model.add(pooling(pool_size=(2, 2), strides=2))
        model.add(Dropout(dropout))
        if batch_norm:
            model.add(BatchNormalization())
        return model

    def model_architecture(self, model_param_dict):
        # dynamically creates variables in the scope of this function
        # using the dict key and assigns the value to that new variable
        # for key, value in model_param_dict.items():
        # exec(f'{key}={value}', globals())

        # regularization variable is generated from above exec statement

        first_layer = model_param_dict["first_filter"]

        print(first_layer)

    def scope_test_origin(self):
        self.scope_test = 5

    def scope_test_dest(self):
        print(self.scope_test * 2)

    def remove_extra_conv_params(self, param_dict: dict, num_conv_layers: int) -> dict:
        """
        Convolution parameters are named using ordinals. To maintain naming,
        this function stores needed conv params and removes any not needed.

        Requires num_conv_layers <=10. Due to the tight coupling with
        params.yaml, python libraries that handle integer/ordinal word
        conversion were not implemented since params may not match the return.

        Note: Since the cartesean product is used to create a parameter space
        this method prevents unnecessary model training by removing any conv
        parameter not needed based on the number of conv layers to model

        Example: third_kernel size is not needed when two convolution layers
        are trained, however if not removed from the parameter dict before
        permutation, it will bloat the list of models to be trained costing
        time and compute resources.
        """

        ordinal_list = [
            "first",
            "second",
            "third",
            "forth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "nineth",
            "tenth",
        ]

        ordinals_to_remove = ordinal_list[num_conv_layers:]

        keys_to_remove = []
        for key in param_dict.keys():
            for ordinal in ordinals_to_remove:
                if ordinal in key:
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del param_dict[key]

        return param_dict

    def permute_model_parameters(self):
        """
        Creates cartesian product of model parameters from params.yaml

        Each param dict in this list will trained and evaluated.
        """

        num_conv_layers_list = self.params.model_training.global_params.num_conv_layers

        all_permuted_parameters = []
        for num_conv_layers in num_conv_layers_list:
            param_dict = dict(self.params.model_training.model_params)
            param_dict = self.remove_extra_conv_params(param_dict, num_conv_layers)

            permuted_parameters = [
                dict(zip(param_dict, v)) for v in product(*param_dict.values())
            ]
            all_permuted_parameters += permuted_parameters

        return all_permuted_parameters
