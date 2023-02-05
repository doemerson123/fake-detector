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
    return ConfigBox(params_dict)


def mock_num_conv_layers():
    num_conv_layers = {num_conv_layers: [2]}
    return ConfigBox(num_conv_layers)


@pytest.fixture()
def load_test_params():
    return load_params(path.join("test", "test_params.yaml"))


@pytest.fixture()
def test_image_dir():
    src_path = Path(__file__).parent.parent.parent.resolve()
    test_image_dir = path.join(
        src_path, "fake-detector", "test", "Fake", "test_Fake 4.jpg"
    )
    return test_image_dir


@pytest.fixture()
def batch_size():
    params = load_params(path.join("test", "test_params.yaml"))
    return params.model_training.model_params.batch_size[0]


@pytest.fixture()
def img_size():
    params = load_params(path.join("test", "test_params.yaml"))
    return params.model_training.global_params.img_size


@pytest.fixture()
def one_layer_removed_params_dict(mock_params_dict):
    return mt.remove_extra_conv_params(mock_params_dict, 2)


@pytest.fixture()
def mock_model(mock_params_dict):

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
    mock_model.build()
    model_json = json.loads(mock_model.to_json())
    top_layer = json_normalize(model_json)
    return json_normalize(top_layer["config.layers"][0])


@pytest.fixture()
def first_conv_layer_input_attributes(mock_model_attributes):
    return mock_model_attributes.iloc[0, :]


@pytest.fixture()
def first_conv_layer_conv_layer_attributes(mock_model_attributes):
    return mock_model_attributes.iloc[1, :]


@pytest.fixture()
def first_conv_layer_pooling_attributes(mock_model_attributes):
    return mock_model_attributes.iloc[2, :]


@pytest.fixture()
def first_conv_layer_dropout_attributes(mock_model_attributes):
    return mock_model_attributes.iloc[3, :]


@pytest.fixture()
def first_conv_layer_batch_norm_attributes(mock_model_attributes):
    return mock_model_attributes.iloc[4, :]


@pytest.fixture()
def second_conv_layer_conv_layer_attributes(mock_model_attributes):
    return mock_model_attributes.iloc[5, :]


@pytest.fixture()
def second_conv_layer_pooling_attributes(mock_model_attributes):
    return mock_model_attributes.iloc[6, :]


@pytest.fixture()
def second_conv_layer_dropout_attributes(mock_model_attributes):
    return mock_model_attributes.iloc[7, :]


@pytest.fixture()
def second_conv_layer_batch_norm_attributes(mock_model_attributes):
    return mock_model_attributes.iloc[8, :]


@pytest.fixture()  ###########
def flatten_attributes(mock_model_attributes):
    return mock_model_attributes.iloc[9, :]


@pytest.fixture()
def dense_layer_batch_norm_attributes(mock_model_attributes):
    return mock_model_attributes.iloc[10, :]


@pytest.fixture()
def dense_layer_attributes(mock_model_attributes):
    return mock_model_attributes.iloc[11, :]
