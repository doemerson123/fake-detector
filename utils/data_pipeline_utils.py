import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from imutils import paths
import yaml
from box import ConfigBox


def load_params(params_file: str) -> ConfigBox:
    '''
    Loads parameter file in root directory of project {params.yaml}
    '''

    src_path = Path(__file__).parent.parent.resolve()
    params_file = src_path.joinpath(params_file)
    try:
        with open(params_file, "r") as file:
            params = yaml.safe_load(file)
    except:
        print("ERROR: check that param.yaml exists and is named correctly")
    finally:
        file.close()

    return ConfigBox(params)

def filepath(directory_type:str) -> str:
    '''
    Returns filepath and correct slash for the filesystem based on params.yaml

    directory_type ['data', 'callback', 'artifact', 'root']
    '''
    
    training_locally_bool = params.root_directory.training_locally_bool

    if training_locally_bool:
        root_directory = params.root_directory.local_filepath
        slash = params.root_directory.local_slash
    else:
        root_directory = params.root_directory.cloud_filepath
        slash = params.root_directory.cloud_slash

    if directory_type == 'data':
        return root_directory + "Data", slash
    if directory_type == 'callback':
        return root_directory + "Callbacks", slash
    if directory_type == 'artifact':
        return root_directory + "Artifacts", slash
    if directory_type == 'root':
        return root_directory, slash


params = load_params('params.yaml')
img_size = params.model_training.model_params.img_size

def load_images(image_path: filepath) -> tuple(tf.const, tf.const):
    '''
    Converts jpeg file (X) and class (Y) into tf.const

    Returns tuple of the image and label tensors
    '''

    # read the image from disk, decodes, converts to tensor, and sizes
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (img_size, img_size))

    # parse the class label from the file path
    label = tf.strings.split(image_path, os.path.sep)[-2]
    if label == 'Fake':
        value = tf.constant([1.,0.])
    elif label == 'Real':
        value = tf.constant([0.,1.])

    # return the image and the label tensors as tuple
    return (image, value)

@tf.function
def augment(image:tf.constant, label:tf.constant) -> tuple:
    '''
    Performs random perterbations of training images using random numpy states.
    Action performed at batch level as Tensorflow dataset reads from disk

    Tensorflow datasets required label to pass through - variable is unchanged
    '''

    # choose random integer
    rand_int = np.random.randint(1,1000)

    # update numpy pseudo-random state to promote true randomness
    # numpy state is preserved globally - next function call will use new state 
    np.random.seed(rand_int)

    # mod of the rand_int can satisfy multiple cases 
    if rand_int % 2==0:
        image = tf.image.random_hue(image, 0.04)
    if rand_int % 3==0:
        image = tf.image.random_saturation(image, 0.8, 1.2)
    if rand_int % 5==0:
        image = tf.image.random_brightness(image, 0.05)
    if rand_int % 12==0:
        image = tf.image.random_contrast(image, 0.9, 1.1)

    # tensorflow's random flip along vertical axis (50% probability of flip)
    image = tf.image.random_flip_left_right(image)

    return (image, label)

def train_test_val_filepaths() -> tuple(str, str, str):
    '''
    Returns filepaths for train, test, and validation data directories 
    '''

    root_directory, slash = filepath('data')
    root_directory += slash

    train_filepath = list(paths.list_images(root_directory + 'Train' + slash))
    val_filepath = list(paths.list_images(root_directory + 'Val' + slash))
    test_filepath = list(paths.list_images(root_directory + 'Test' + slash))
    return train_filepath, val_filepath, test_filepath

def define_dataset_generator(filepath:str, cache_name:str, batch_size:int) -> tf.dataset:
    '''
    Defines parameters for dataset generators
    '''
    
    # create dataset generator object
    dataset = tf.data.Dataset.from_tensor_slices(filepath)
    
    # define parameters for dataset generators 
    if 'Train' in filepath:
        # filepath needed to calculate the lenth for shuffle - only for train
        train_filepath, _, _ = train_test_val_filepaths()    
        dataset = (dataset
                .shuffle(len(train_filepath))
                .map(load_images, num_parallel_calls = tf.data.experimental.AUTOTUNE)
                .map(augment, num_parallel_calls = tf.data.experimental.AUTOTUNE)
                .cache(cache_name)
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
                .repeat(1)
                )
    else:
        dataset = (dataset
                .map(load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .cache(cache_name)
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
                )

    return dataset

def datasets(batch_size):
    '''
    Creates datasets for model training 

    batch_size = number of images to retreive from disk - int
    '''
    train_filepath, val_filepath, test_filepath = data_filepaths()

    # build dataset generators. Note train data is randomized,
    # perturbed, and is read into the train set twice with diferent pertubations
    train_dataset = define_dataset_generator(train_filepath, 'train_cache', batch_size)
    val_dataset = define_dataset_generator(val_filepath, 'val_cache', batch_size)
    test_dataset = define_dataset_generator(test_filepath, 'test_cache', batch_size)

    return train_dataset, val_dataset, test_dataset



