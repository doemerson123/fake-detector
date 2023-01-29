from __future__ import annotations
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from imutils import paths
import yaml
from box import ConfigBox
from typing import Tuple, List


def load_params(params_file: str = "params.yaml") -> ConfigBox:
    """
    Loads parameter file from root directory of project {params.yaml}
    """

    src_path = Path(__file__).parent.parent.resolve()
    params_file = src_path.joinpath(params_file)
    try:
        with open(params_file, "r") as file:
            params = yaml.safe_load(file)
    except:
        print("ERROR: check that params.yaml exists and is named correctly")
    finally:
        file.close()

    return ConfigBox(params)


params = load_params()
img_size = params.model_training.global_params.img_size


def file_directory(directory_type: str) -> str:
    """
    Returns filepath and correct slash for the filesystem based on params.yaml

    directory_type {'data', 'callback', 'artifact', 'root'}
    """

    training_locally_bool = params.root_directory.training_locally_bool

    if training_locally_bool:
        root_directory = params.root_directory.local_filepath
    else:
        root_directory = params.root_directory.cloud_filepath

    if directory_type == "data":
        # return root_directory + "Data", slash
        return os.path.join(root_directory, "Data")
    if directory_type == "callback":
        return os.path.join(root_directory, "Callbacks")
    if directory_type == "artifact":
        return os.path.join(root_directory, "Artifacts")
    if directory_type == "root":
        return root_directory


def load_image(image_path: str) -> Tuple[tf.constant, tf.constant]:
    """
    Converts both image file (X) and class (Y) into a tuple of tf.constant
    """

    # read the image from disk, decodes, converts to tensor, and resizes
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (img_size, img_size))

    # parse the class label from the file path
    label = tf.strings.split(image_path, os.path.sep)[-2]
    if label == "Fake":
        value = tf.constant([1.0, 0.0])
    if label == "Real":
        value = tf.constant([0.0, 1.0])

    # return the image and the label tensors as tuple
    return (image, value)


@tf.function
def augment(image: tf.constant, label: tf.constant) -> tuple(tf.constant, tf.constant):
    """
    Performs random perterbations of training images using random numpy states.
    Action performed on each image as it's read from disk

    Tensorflow datasets required label to pass through, its value is unchanged
    """

    # choose random integer
    rand_int = np.random.randint(1, 1000)

    # update numpy pseudo-random state to promote true randomness
    # numpy state is preserved globally - next function call will use new state
    np.random.seed(rand_int)

    # mod of the rand_int can satisfy multiple cases
    if rand_int % 2 == 0:
        image = tf.image.random_hue(image, 0.04)
    if rand_int % 3 == 0:
        image = tf.image.random_saturation(image, 0.8, 1.2)
    if rand_int % 5 == 0:
        image = tf.image.random_brightness(image, 0.05)
    if rand_int % 12 == 0:
        image = tf.image.random_contrast(image, 0.9, 1.1)

    # tensorflow standard function - 50% probability of flip on vertical axis
    image = tf.image.random_flip_left_right(image)

    return (image, label)


def train_test_val_filepaths() -> tuple(List[str], List[str], List[str]):
    """
    Returns filepaths for train, test, and validation data directories
    """

    root_directory, slash = file_directory("root")
    root_directory += slash

    train_filepath = list(paths.list_images(root_directory + "Train" + slash))
    val_filepath = list(paths.list_images(root_directory + "Val" + slash))
    test_filepath = list(paths.list_images(root_directory + "Test" + slash))
    return train_filepath, val_filepath, test_filepath


def dataset_generator(filepath: str, batch_size: int, cache_name: str) -> tf.dataset:
    """
    Defines dataset generators used to create batches for model training.

    Train data is shuffled, perturbed, and read into the training dataset
    twice with diferent pertubations each time using .repeat(1)

    Validation and test data are not modified.
    """

    # create dataset generator object
    dataset = tf.data.Dataset.from_tensor_slices(filepath)

    # define parameters for dataset generators
    if "Train" in filepath:
        # filepath needed to calculate the lenth for shuffle - only for train
        train_filepath, _, _ = train_test_val_filepaths()
        dataset = (
            dataset.shuffle(len(train_filepath))
            .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .cache(cache_name)
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
            .repeat(1)
        )
    else:
        dataset = (
            dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .cache(cache_name)
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    return dataset


def create_dataset(dataset_type: str, batch_size: int) -> tf.dataset:
    """
    Generates required dataset {train, val, test}

    Reminder: train data is randomized, perturbed, and read into
    the train dataset twice with diferent pertubations

    batch_size = number of images to retreive from disk - int
    """

    train_filepath, val_filepath, test_filepath = train_test_val_filepaths()
    if dataset_type == "train":
        return dataset_generator(train_filepath, batch_size, "train_cache")
    if dataset_type == "val":
        return dataset_generator(val_filepath, batch_size, "val_cache")
    if dataset_type == "test":
        return dataset_generator(test_filepath, batch_size, "test_cache")
