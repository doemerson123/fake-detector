from box import ConfigBox
from tensorflow.keras.models import Sequential
import pytest
from tensorflow.keras.layers import AvgPool2D, Flatten, Dense
import tensorflow as tf
from utils.data_pipeline_utils import load_params
from pathlib import Path
from os import path
from utils.modeling_utils import ModelTraining
from pandas import DataFrame, json_normalize
import json

mt = ModelTraining(params_file=path.join("test", "test_params.yaml"))


@pytest.fixture()
def mock_params_dict():
    params_dict = {
        "dense_layers": 1,
        "first_filter": 10,
        "first_kernel": 3,
        "second_filter": 15,
        "second_kernel": 5,
        "third_filter": 17,
        "third_kernel": 7,
        "batch_size": 100,
        "beta_1": 0.5,
        "pooling": "AvgPool2D",
        "dense_nodes": 10,
        "learning_rate": 0.002,
        "regularization": [True, 0.002, 0.40],
    }
    """
    Pytest fixture to return the dictionary containing the model parameters.

    Returns
    -------
        ConfigBox: The dictionary containing the model parameters.
        """
    return ConfigBox(params_dict)


def mock_num_conv_layers():
    """
    Returns the number of convolutional layers of a model in the form of a
    ConfigBox.

    Returns
    -------
        ConfigBox: A ConfigBox containing the number of convolutional layers of
        a model.
    """
    num_conv_layers = {num_conv_layers: [2]}
    return ConfigBox(num_conv_layers)


@pytest.fixture()
def load_test_params():
    """
    Pytest fixture that loads the model parameters from a YAML file.

    Returns
    -------
        object: The model parameters loaded from a YAML file.
    """
    return load_params(path.join("test", "test_params.yaml"))


@pytest.fixture()
def test_image_dir():
    """
    Pytest fixture that returns the directory path of a test image.

    Returns
    -------
        str: The directory path of a test image.
    """
    src_path = Path(__file__).parent.parent.parent.resolve()
    test_image_dir = path.join(
        src_path, "fake-detector", "test", "Fake", "test_Fake 4.jpg"
    )
    return test_image_dir


@pytest.fixture()
def batch_size():
    """
    A fixture to retrieve the batch size from the loaded model parameters.

    Returns
    -------
        int: The batch size
    """
    params = load_params(path.join("test", "test_params.yaml"))
    return params.model_training.model_params.batch_size[0]


@pytest.fixture()
def img_size():
    """
    A fixture to retrieve the image size from the loaded model parameters.

    Returns
    -------
        int: The image size
    """
    params = load_params(path.join("test", "test_params.yaml"))
    return params.model_training.global_params.img_size


@pytest.fixture()
def one_layer_removed_params_dict(mock_params_dict):
    """
    A fixture to retrieve the model parameters with one layer removed.

    Parameters
    ----------
        mock_params_dict dict
            The model parameters as a dictionary

    Returns
    -------
        dict: The model parameters with one layer removed
    """
    return mt.remove_extra_conv_params(mock_params_dict, 2)


@pytest.fixture()
def mock_model(mock_params_dict):
    """
    A fixture to generate a mock model using the provided model parameters.

    Parameters
    ----------
        mock_params_dict (dict): The model parameters as a dictionary

    Returns
    -------
        model: Sequential object
            The mock model generated from the provided parameters
    """

    model = Sequential()
    model = mt.conv_layer(
        model, mock_params_dict, filters=10, kernel_size=3, first_layer_bool=True
    )
    model = mt.conv_layer(
        model, mock_params_dict, filters=15, kernel_size=5, first_layer_bool=False
    )
    model.add(Flatten())
    model = mt.dense_layer(model, mock_params_dict)
    model.add(Dense(units=2, activation=tf.nn.softmax))
    return model


@pytest.fixture()
def mock_model_attributes(mock_model):
    """
    A fixture to retrieve the attributes of the mock model.

    Parameters
    ----------
        mock_model (Sequential object): The mock model

    Returns
    -------
        dataframe: The attributes of the mock model as a dataframe
    """
    mock_model.build()
    model_json = json.loads(mock_model.to_json())
    top_layer = json_normalize(model_json)
    return json_normalize(top_layer["config.layers"][0])


@pytest.fixture()
def first_conv_layer_input_attributes(mock_model_attributes):
    """
    A fixture to retrieve the input attributes of the first convolutional layer
    of the mock model.

    Parameters
        ----------
        mock_model_attributes (pandas.DataFrame): A pandas dataframe with all 
        the attributes of the model

    Returns
    -------
        dataframe: The input attributes of the first convolutional layer of the
        mock model as a dataframe
    """
    return mock_model_attributes.iloc[0, :]


@pytest.fixture()
def first_conv_layer_conv_layer_attributes(mock_model_attributes):
    """
    A fixture to retrieve the attributes of the first convolutional layer of
    the model.

    Parameters
        ----------
        mock_model_attributes (pandas.DataFrame): A pandas dataframe with all 
        the attributes of the model

    Returns
    -------
        pandas.Series: The attributes of the first convolutional layer.
    """
    return mock_model_attributes.iloc[1, :]


@pytest.fixture()
def first_conv_layer_pooling_attributes(mock_model_attributes):
    """
    A fixture to retrieve the attributes of the pooling layer following the
    first convolutional layer of the model.

    Parameters
        ----------
        mock_model_attributes (pandas.DataFrame): A pandas dataframe with all 
        the attributes of the model

    Returns
    -------
        pandas.Series: The attributes of the pooling layer following the first
        convolutional layer.
    """
    return mock_model_attributes.iloc[2, :]


@pytest.fixture()
def first_conv_layer_dropout_attributes(mock_model_attributes):
    """
    A fixture to retrieve the attributes of the dropout layer following the
    first convolutional layer of the model.

    Parameters
        ----------
        mock_model_attributes (pandas.DataFrame): A pandas dataframe with all 
        the attributes of the model

    Returns
    -------
        pandas.Series: The attributes of the dropout layer following the first
        convolutional layer.
    """

    return mock_model_attributes.iloc[3, :]


@pytest.fixture()
def first_conv_layer_batch_norm_attributes(mock_model_attributes):
    """
    A fixture to retrieve the attributes of the batch normalization layer
    following the first convolutional layer of the model.

    Parameters
        ----------
        mock_model_attributes (pandas.DataFrame): A pandas dataframe with all 
        the attributes of the model

    Returns
    -------
        pandas.Series: The attributes of the batch normalization layer
        following the first convolutional layer.
    """
    return mock_model_attributes.iloc[4, :]


@pytest.fixture()
def second_conv_layer_conv_layer_attributes(mock_model_attributes):
    """
    A fixture to retrieve the attributes of the second convolutional layer of
    the model.

    Parameters
        ----------
        mock_model_attributes (pandas.DataFrame): A pandas dataframe with all 
        the attributes of the model

    Returns
    -------
        pandas.Series: The attributes of the second convolutional layer.
    """
    return mock_model_attributes.iloc[5, :]


@pytest.fixture()
def second_conv_layer_pooling_attributes(mock_model_attributes):
    """
    A fixture to retrieve the attributes of the pooling layer following the
    second convolutional layer of the model.

    Parameters
        ----------
        mock_model_attributes (pandas.DataFrame): A pandas dataframe with all 
        the attributes of the model

    Returns
    -------
        pandas.Series: The attributes of the pooling layer following the second
        convolutional layer.
    """
    return mock_model_attributes.iloc[6, :]


@pytest.fixture()
def second_conv_layer_dropout_attributes(mock_model_attributes):
    """
    A fixture to retrieve the attributes of the dropout layer following the
    second convolutional layer of the model.

    Parameters
        ----------
        mock_model_attributes (pandas.DataFrame): A pandas dataframe with all 
        the attributes of the model

    Returns
    -------
        pandas.Series: The attributes of the dropout layer following the second
        convolutional layer.
    """
    return mock_model_attributes.iloc[7, :]


@pytest.fixture()
def second_conv_layer_batch_norm_attributes(mock_model_attributes):
    """
    A fixture to retrieve the attributes of the second convolutional layer's 
    batch normalization layer from the loaded model parameters.

    Parameters
    ----------
    mock_model_attributes (pandas.DataFrame): A pandas dataframe with all the 
    attributes of the model

    Returns
    -------
        pandas.Series: The attributes of the second convolutional layer's batch
        normalization layer
    """
    return mock_model_attributes.iloc[8, :]


@pytest.fixture()  ###########
def flatten_attributes(mock_model_attributes):
    """
    A fixture to retrieve the attributes of the flatten layer from the loaded
    model parameters.
    
    Parameters
    ----------
    mock_model_attributes (pandas.DataFrame): A pandas dataframe with all the 
    attributes of the model
    
    Returns
    -------
        pandas.Series: The attributes of the flatten layer
    """
    return mock_model_attributes.iloc[9, :]


@pytest.fixture()
def dense_layer_batch_norm_attributes(mock_model_attributes):
    """
    A fixture to retrieve the attributes of the dense layer's batch 
    normalization layer from the loaded model parameters.
    
    Parameters
    ----------
        mock_model_attributes (pandas.DataFrame): A pandas dataframe with all 
        the attributes of the model
    
    Returns
    -------
        pandas.Series: The attributes of the flatten layer
    """
    return mock_model_attributes.iloc[10, :]


@pytest.fixture()
def dense_layer_attributes(mock_model_attributes):
    """
    A fixture to retrieve the attributes of the dense layer from the loaded
    model parameters.
    
    Parameters
    ----------
        mock_model_attributes (pandas.DataFrame): A pandas dataframe with all 
        the attributes of the model
    
    Returns
    -------
        pandas.Series: The attributes of the flatten layer
    """
    return mock_model_attributes.iloc[11, :]
