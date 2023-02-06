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
    Check if the training is being done locally. Used to drive OS specific 
    behavior.
    
    Returns
    -------
    bool
        True if training is being done locally, False otherwise.
    """
    if params.root_directory.training_locally_bool:
        return True
    else:
        return False


def load_params(params_file: str = "params.yaml") -> ConfigBox:
    """
    Load parameters from a YAML file and return as a ConfigBox object to allow
    for dot notation.
    
    Parameters
    ----------
    params_file : str, optional
        Path to the YAML file containing the parameters, by default "params.yaml".
    
    Returns
    -------
    ConfigBox
        Object containing the parameters from the YAML file.
    
    Raises
    ------
    FileNotFoundError
        If the specified YAML file is not found.
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

    Parameters:
    ----------
        directory_type (str): directory type to get filepath for 
            {'data', 'callback', 'artifact', 'root'}

    Returns:
    --------
        str: os specific filepath for the selected directory type

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
    Loads an image from the specified file path and returns it along with its
    label as a tuple of tensors.

    Parameters:
    ----------
        image_path (str): The file path of the image.

    Returns:
    --------
        Tuple[tf.constant, tf.constant]: A tuple of two tensors, the first 
        representing the image and the second representing the label of the 
        image. The image is a 3 channel tensor of dtype tf.float32 and the 
        label is a 2 element tensor of dtype tf.constant. The label represents 
        the class of the image, either 'Fake' or 'Real', with [1.0, 0.0] and 
        [0.0, 1.0] respectively.
    """

    # read the image from disk, decodes, converts to tensor, and resizes
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (img_size, img_size))

    # parse the class label from the file path
    label = tf.strings.split(image_path, os.path.sep)[-2]

    # tf equal requires functions to evaluate the equality
    def fake_label():
        return tf.constant([1.0, 0.0])

    def real_label():
        return tf.constant([0.0, 1.0])

    value = tf.cond(
        tf.math.equal(label, tf.constant(["Fake"], dtype=tf.string)),
        fake_label,
        real_label,
    )

    # return the image and the label tensors as tuple
    return (image, value)


@tf.function
def augment(image: tf.constant, label: tf.constant) -> tuple(tf.constant, tf.constant):
    """
    Custom tensorflow helper function used in creating datasets.

    The function performs random changes to the image, including random hue, 
    saturation, brightness, contrast changes, and a random flip on the vertical
    axis.

    The mod of rand_int is used to determine which alterations an image will
    receive. Zero, one, or many alterations are possible..

    Tensorflow datasets require label in the signature but no changes are made
    as it passes thorugh this function.

    Randomly augment an image and its label.
    

    Parameters
    ----------
        image (tf.constant): The image to be augmented.
        label (tf.constant): The label associated with the image.
    
    Returns
    -------
    tuple(tf.constant, tf.constant)
        The augmented image and its label.
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

    Returns
    -------
    tuple(List[str], List[str], List[str])
        List of file paths for the train set, validation set, and test set.
    """

    data_directory = file_directory("data")

    train_filepaths = list(paths.list_images(os.path.join(data_directory, "Train")))
    val_filepaths = list(paths.list_images(os.path.join(data_directory, "Val")))
    test_filepaths = list(paths.list_images(os.path.join(data_directory, "Test")))
    return train_filepaths, val_filepaths, test_filepaths


def dataset_generator(filepath: str, batch_size: int, cache_name: str) -> tf.dataset:
    """
    Defines dataset generators used to create batches for model training.

    Notes:
    - If the filepath contains 'Train', the dataset is shuffled, augmented for 
    training, and read into the dataset TWICE.
    - Dataset is unmodified for validation or testing.
    - The first time the dataset is created, a cache is saved with the 
    specified `cache_name` so that subsequent calls to the function will reuse 
    the cache. This saves time and compute resources plus ensures all
    experiments are conducted on the same underlying data.

    Parameters
    ----------
        filepath (str): path to the file with data
        batch_size (int): size of the batch to return in each iteration
        cache_name (str): name to use for caching the data

    Returns:
    ----------
    tf.dataset: a TensorFlow dataset 

    
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
    Generates required dataset when provided string {'train', 'val', 'test'}

    Reminder: train data is randomized, perturbed, and read into
    the train dataset twice with diferent pertubations then cached after the
    first read from disk
    
    Parameters
    ----------
        dataset_type (str): type of dataset to generate {'train', 'val', 'test'}
        batch_size (int): number of images to retreive from disk 
    
    Returns:
    ----------
    tf.dataset: a TensorFlow dataset 
    """

    train_filepath, val_filepath, test_filepath = train_test_val_filepaths()
    if dataset_type == "train":
        return dataset_generator(train_filepath, batch_size, "train_cache")
    if dataset_type == "val":
        return dataset_generator(val_filepath, batch_size, "val_cache")
    if dataset_type == "test":
        return dataset_generator(test_filepath, batch_size, "test_cache")
