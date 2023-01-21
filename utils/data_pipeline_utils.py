import tensorflow as tf
import numpy as np
from imutils import paths
import time
import os
import pandas as pd
import yaml
from box import ConfigBox


def load_params():
    root_directory, slash = filepath('root')
    params_filepath = root_directory + slash + 'params.yaml'
    #params_file = src_path.joinpath(params_filepath)

    with open(params_filepath, "r") as f:
        params = yaml.safe_load(f)
        params = ConfigBox(params)
    return params

def filepath(directory_type):
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
        return root_directory + slash + "Data", slash
    elif directory_type == 'callback':
        return root_directory + slash + "Callbacks", slash
    elif directory_type == 'artifact':
        return root_directory + slash + "Artifacts", slash
    elif directory_type == 'root':
        return root_directory, slash


params = load_params()

img_size = params.model_training.model_params.img_size

def load_images(imagePath):
    '''
    Converts jpeg file (X) and class (Y) into tensors
    '''

    # read the image from disk, decodes, converts to tensor, and sizes
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (img_size, img_size))

    # parse the class label from the file path
    label = tf.strings.split(imagePath, os.path.sep)[-2]
    if label == 'Fake':
        value = tf.constant([1.,0.])
    elif label == 'Real':
        value = tf.constant([0.,1.])

    # return the image and the label tensors as tuple
    return (image, value)

@tf.function
def augment(image, label):
    # perform random horizontal and vertical flips
    rando = np.random.randint(1,1000)
    
    image = tf.image.random_flip_left_right(image)
    if rando%2==0: 
        image = tf.image.random_hue(image, 0.04) 
    if rando%3==0:
        image= tf.image.random_saturation(image, 0.8, 1.2) 
    if rando%5==0:
        image = tf.image.random_brightness(image, 0.05) 
    if rando%12==0:
        image = tf.image.random_contrast(image, 0.9, 1.1) 
    # return the image and the label
    return (image, label)




def train_test_val_filepaths():

    filepath, slash = filepath('data')
    train_filepath = list(paths.list_images(filepath + 'Train' + slash))
    val_filepath = list(paths.list_images(filepath + 'Val' + slash))
    test_filepath = list(paths.list_images(filepath + 'Test' + slash))
    return train_filepath, val_filepath, test_filepath

def define_dataset_generator(filepath, cache_name, batch_size):
    '''
    Defines parameters for dataset generators


    filepath = from_tensor_slices - object
    cache_name = name of cache to use - string
    batch_size = number of images in batch - int
    
    '''
    
    # create dataset generator object
    dataset = tf.data.Dataset.from_tensor_slices(filepath)
    
    # define parameters for dataset generators 
    if 'Train' in filepath:
        # filepath needed to calculate the lenth for shuffle - only for train
        train_filepath, _, _ = data_filepaths()    
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
    # perturbed, and is read into the train set twice with different pertubations
    train_dataset = define_dataset_generator(train_filepath, 'train_cache', batch_size)
    val_dataset = define_dataset_generator(val_filepath, 'val_cache', batch_size)
    test_dataset = define_dataset_generator(test_filepath, 'test_cache', batch_size)

    return train_dataset, val_dataset, test_dataset



