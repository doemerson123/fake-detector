from __future__ import annotations
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from imutils import paths
import yaml
from box import ConfigBox
from typing import Tuple, List


def training_locally():
    """
    Returns training locally bool from params.yaml which is used to drive os
    specific behavior for file storage and retreival
    """
    if params.root_directory.training_locally_bool:
        return True
    else:
        return False


def load_params(params_file: str = "params.yaml") -> ConfigBox:
    """
    Converts parameter file {params.yaml} from yaml to ConfigBox in order to
    use dot notation. Accepts other .yaml files - used in testing.
    """

    src_path = Path(__file__).parent.parent.resolve()
    params_file = src_path.joinpath(params_file)
    try:
        with open(params_file, "r") as file:
            params = yaml.safe_load(file)
            file.close()
            return ConfigBox(params)
    except OSError:
        raise FileNotFoundError("Unable to load params.yaml, check location and name")


params = load_params()
img_size = params.model_training.global_params.img_size


def file_directory(directory_type: str) -> str:
    """
    Returns os specific filepath for the filesystem based root_directory
    parameters in params.yaml


    directory_type {'data', 'callback', 'artifact', 'root'}
    """

    if training_locally():
        root_directory = params.root_directory.local_filepath
    else:
        root_directory = params.root_directory.cloud_filepath

    if directory_type == "data":
        return os.path.join(root_directory, "Data")
    if directory_type == "callback":
        return os.path.join(root_directory, "Callbacks")
    if directory_type == "artifact":
        return os.path.join(root_directory, "Artifacts")
    if directory_type == "root":
        return root_directory


@tf.function
def load_image(image_path: str) -> Tuple[tf.constant, tf.constant]:
    """
    Custom tensorflow helper function used in creating datasets.

    Reads jpeg files and converts to tf.float32 for processing.

    Combines image file (X) and class (Y) into a tuple of tf.constant.
    """

    # read the image from disk, decodes, converts to tensor, and resizes
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (img_size, img_size))

    # parse the class label from the file path
    label = tf.strings.split(image_path, os.path.sep)[-2]

    def fake_label():
        return tf.constant([1.0, 0.0])

    def real_label():
        return tf.constant([0.0, 1.0])

    value = tf.cond(
        tf.math.equal(label, tf.constant(["Fake"], dtype=tf.string)),
        fake_label,
        real_label,
    )

    # if label == "Fake":
    #    value = tf.constant([1.0, 0.0])
    # if label == "Real":
    #    value = tf.constant([0.0, 1.0])

    # return the image and the label tensors as tuple
    return (image, value)


@tf.function
def augment(image: tf.constant, label: tf.constant) -> tuple(tf.constant, tf.constant):
    """
    Custom tensorflow helper function used in creating datasets.

    Performs random perterbations of training images using pseudo-random numpy
    states. Action performed on each image as it's read from disk.

    The mod of rand_int is used to determine which alterations an image will
    receive. Zero, one, or many alterations are possible for each image.

    Tensorflow datasets require label in the signature but no changes are made
    as it passes thorugh this function.
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
    Returns filepaths for train, test, and validation data directories.

    Requires directory tree: Data/{Train, Val, Test}/{Class 1, Class 2}
    """

    data_directory = file_directory("data")

    train_filepaths = list(paths.list_images(os.path.join(data_directory, "Train")))
    val_filepaths = list(paths.list_images(os.path.join(data_directory, "Val")))
    test_filepaths = list(paths.list_images(os.path.join(data_directory, "Test")))
    return train_filepaths, val_filepaths, test_filepaths


def dataset_generator(filepath: str, batch_size: int, cache_name: str) -> tf.dataset:
    """
    Defines dataset generators used to create batches for model training.

    Note: train data is shuffled, perturbed, and read into the training dataset
    a second time with diferent alterations using .repeat(1)

    Note: The first time data is read in a cache is created so each model that
    is trained uses the same datasets. Important because the train set is
    heavily augmented and read twice

    Validation and test data are not modified and simply read from disk
    """
    dataset = tf.data.Dataset.from_tensor_slices(filepath)

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
    the train dataset twice with diferent pertubations then cached after the
    first read from disk

    batch_size = number of images to retreive from disk - int
    """

    train_filepath, val_filepath, test_filepath = train_test_val_filepaths()
    if dataset_type == "train":
        return dataset_generator(train_filepath, batch_size, "train_cache")
    if dataset_type == "val":
        return dataset_generator(val_filepath, batch_size, "val_cache")
    if dataset_type == "test":
        return dataset_generator(test_filepath, batch_size, "test_cache")
