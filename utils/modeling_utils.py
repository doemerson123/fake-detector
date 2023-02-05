import time
from itertools import product
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Dropout,
    Conv2D,
    Dense,
    Flatten,
    MaxPool2D,
    AvgPool2D,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from utils.custom_metrics_utils import StatefullMultiClassFBeta
from utils.plot_metrics_utils import (
    save_performance_artifacts,
    rounded_evaluate_metrics,
)
from utils.data_pipeline_utils import load_params
from utils.data_pipeline_utils import create_dataset, file_directory
from typing import Dict, List


class ModelTraining:
    """
    Defines parameter space, creates model architecture, and performs
    experiment tracking.
    """

    def __init__(self, params_file="params.yaml") -> None:
        self.params = load_params(params_file)
        self.img_size = self.params.model_training.global_params.img_size
        self.epochs = self.params.model_training.global_params.max_epochs

    def remove_extra_conv_params(self, param_dict: dict, num_conv_layers: int) -> Dict:
        """
        Convolution parameters are named using ordinals in params.yaml. To
        maintain the naming convention, this function removes any ordinal named
        conv paramters that are not needed based on num_conv_layers.

        Example: third_kernel and third_filter are not needed when
        two convolution layers are trained. If not removed before permutation,
        these params will geometrically bloat the list of param_dicts - driving
        considerable time and compute cost.

        Note: Requires num_conv_layers <=10. Due to the tight coupling with
        params.yaml, python libraries that handle integer/ordinal word
        conversion were not implemented to intentionally force users to update
        params.yaml and this code if extended beyond 10 convolution layers.
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

    def permute_model_parameters(self) -> List:
        """
        Creates cartesian product of model parameters from params.yaml by
        looping through the the list of num_conv_layers.

        Each dict in the returned list will trained and evaluated.
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

    def conv_layer(
        self,
        model: Sequential,
        params: dict,
        filters: int,
        kernel_size: int,
        first_layer_bool: bool,
    ) -> Sequential:
        """
        Creates convolution layer using a single set of params.
        """

        batch_norm = params["regularization"][0]
        dropout = params["regularization"][2]

        # first layer requires input parameters
        if first_layer_bool:
            model.add(
                Conv2D(
                    filters=filters,
                    kernel_size=(kernel_size, kernel_size),
                    strides=(1, 1),
                    activation=tf.nn.relu,
                    input_shape=(self.img_size, self.img_size, 3),
                )
            )
        else:
            model.add(
                Conv2D(
                    filters=filters,
                    kernel_size=(kernel_size, kernel_size),
                    strides=(1, 1),
                    activation=tf.nn.relu,
                )
            )

        model.add(eval(params["pooling"])(pool_size=(2, 2), strides=2))
        model.add(Dropout(dropout))

        if batch_norm:
            model.add(BatchNormalization())

        return model

    def dense_layer(self, model: Sequential, params: dict) -> Sequential:
        """
        Dynamically creates dense layer using params dict
        """

        batch_norm = params["regularization"][0]
        l2_alpha = params["regularization"][1]

        if batch_norm:
            model.add(BatchNormalization())

        model.add(
            Dense(
                units=params["dense_nodes"],
                activation=tf.nn.relu,
                kernel_regularizer=tf.keras.regularizers.l2(l2_alpha),
            )
        )
        return model

    def model_architecture(self, params: dict, model_name: str) -> Sequential:
        """
        Creates Sequential model using single params dict from the list of
        permuted_model_params
        """

        # custom F1 metric
        statefull_multi_class_fbeta = StatefullMultiClassFBeta()

        optimizer = Nadam(learning_rate=params["learning_rate"], name="Nadam")
        model = Sequential()

        # dynamically create list of ordinal filter/kernel keys to create layers
        # dicts preserve order from python 3.6 onward
        conv_filter_list = []
        conv_kernel_list = []
        for key in params.keys():
            if "filter" in key:
                conv_filter_list.append(key)
            if "kernel" in key:
                conv_kernel_list.append(key)

        # dynamically create convolution layers
        for filter_layer, kernel_layer in zip(conv_filter_list, conv_kernel_list):

            # first layer requires img_size input parameter
            first_layer_bool = False
            if "first" in filter_layer:
                first_layer_bool = True

            model = self.conv_layer(
                model,
                params,
                params[filter_layer],
                params[kernel_layer],
                first_layer_bool,
            )

        # dense layers
        model.add(Flatten())

        dense_layers = params["dense_layers"]
        for dense_layers in range(dense_layers):
            model = self.dense_layer(model, params)
        model.add(Dense(units=2, activation=tf.nn.softmax))

        # compile
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy", statefull_multi_class_fbeta],
        )

        return model

    def train_model(
        self,
        params: dict,
        model_name: str,
    ) -> tuple():

        batch_size = params["batch_size"]

        # the first time data is read in a cache is created so each model that
        # is trained uses the same datasets. Important because the train set is
        # heavily augmented and read twice
        train_dataset = create_dataset("train", batch_size)
        val_dataset = create_dataset("val", batch_size)
        test_dataset = create_dataset("test", batch_size)

        model = self.model_architecture(params, model_name)

        early_stop = EarlyStopping(monitor="val_accuracy", patience=4, verbose=1)
        reduce_learning_rate = ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=2, min_lr=0.00001, verbose=2
        )

        # set model artifact location
        checkpoint_filepath = os.path.join(
            file_directory("artifact"), model_name, model_name, "TF Model"
        )

        if not os.path.isdir(checkpoint_filepath):
            os.makedirs(checkpoint_filepath)

        checkpoint = ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
        )

        hist = model.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=val_dataset,
            callbacks=[early_stop, reduce_learning_rate, checkpoint],
        )

        # Evaluate test dataset and plot
        test_loss, test_accuracy, test_f1 = rounded_evaluate_metrics(
            model, test_dataset, 3
        )
        results_dict = {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
        }

        # callbacks save best model - retreive best validation metric from hist
        # to match this saved model. Find index of best metric, then return all
        best_val_accuracy_index = np.argmax(hist.history["val_state_full_binary_fbeta"])
        for key in hist.history.keys():
            results_dict[key] = hist.history[key][best_val_accuracy_index]

        save_performance_artifacts(model, model_name, hist, test_dataset)

        return hist, results_dict, model

    def train_all_models(self, experiment_name, permuted_model_params):
        """
        Loops through all permuted_model_params and trains each params dict.

        Returns a DataFrame containing performance metrics for all models in
        training run.
        """

        model_list = []
        model_name_list = []
        all_results_dict_list = []
        experiment_counter = int(0)

        for params in permuted_model_params:
            start_time = time.time()

            # store variables for labeling
            experiment_counter += 1
            model_name = f"{experiment_name} {experiment_counter}"

            hist, results_dict, fit_model = self.train_model(params, model_name)

            # legnth of any hist value = number of epochs
            epoch_count = len(hist.history["loss"])

            end_time = time.time()
            experiment_duration = end_time - start_time

            # capture parameters and performance for model in dict to comapre
            # with other trained models once training is completed
            model_parameters_dict = {}
            model_parameters_dict["model_name"] = model_name

            for key in params.keys():
                if key == "regularization":
                    model_parameters_dict["batch_norm"] = params[key][0]
                    model_parameters_dict["l2_alpha"] = params[key][1]
                    model_parameters_dict["dropout"] = params[key][2]
                if key == "pooling":
                    model_parameters_dict[key] = str(params[key])
                else:
                    model_parameters_dict[key] = params[key]

            model_parameters_dict["model_params_count"] = fit_model.count_params()
            model_parameters_dict["experiment_duration"] = experiment_duration
            model_parameters_dict["completed_epochs"] = epoch_count

            # combine model evaluation metrics dict with parameters dict
            model_parameters_dict = {**model_parameters_dict, **results_dict}
            all_results_dict_list.append(model_parameters_dict)

        return pd.DataFrame(all_results_dict_list)
