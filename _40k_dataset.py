import functools
import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import losses
from tensorflow.python.keras import losses
import numpy as np
import os

# # definición de constantes 
# img_shape = (256, 256, 3)  # debe ser divisible por 32 (maxpool2d)
# batch_size = 3
# epochs = 5

def _process_pathnames(fname, label_path):
    # We map this function onto each pathname pair  
    # w x h x 3
    img_str = tf.read_file(fname)
    img = tf.image.decode_jpeg(img_str, channels=3)

    # w x h x 1
    label_img_str = tf.read_file(label_path)
    label_img = tf.image.decode_png(label_img_str, channels=1)
    return img, label_img

# TODO do not waste time resizing, resize the dataset
def _augment(img, label_img, resize=None, scale=1, togray=False):
    """ 
    escalar y "normalizar" imágenes
    """
    if resize is not None:
        label_img = tf.image.resize_images(label_img, resize)
        img = tf.image.resize_images(img, resize)
    
    if togray:
        img = tf.image.rgb_to_grayscale(img)

    # "normalización"
    label_img = tf.to_float(label_img) * scale
    img = tf.to_float(img) * scale 
    return img, label_img

def get_baseline_dataset(filenames, 
                         labels,
                         preproc_fn=functools.partial(_augment),
                         threads=4, 
                         batch_size=3,
                         shuffle=True):           
    num_x = len(filenames)
    # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Map our preprocessing function to every element in our dataset, taking
    # advantage of multithreading
    dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
    dataset = dataset.map(preproc_fn, num_parallel_calls=threads)

    # sloowww https://github.com/tensorflow/tensorflow/issues/14857
    #if shuffle:
    #    dataset = dataset.shuffle(num_x)
    
    # It's necessary to repeat our data for all epochs 
    dataset = dataset.repeat().batch(batch_size)
    return dataset

def read_jpg(path):
    # We map this function onto each pathname pair  
    # w x h x 3
    img_str = tf.read_file(path)
    img = tf.image.decode_jpeg(img_str, channels=3)
    return img

def real_augment(img, resize=None, scale=1,  togray=False):
    if resize is not None:
        img = tf.image.resize_images(img, resize)
    if togray:
        img = tf.image.rgb_to_grayscale(img)
    img = tf.to_float(img) * scale 
    return img

def get_real_dataset(filenames,                          
                         preproc_fn=functools.partial(real_augment),
                         threads=4, 
                         batch_size=3,
                         shuffle=True):           
    num_x = len(filenames)
    # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames))
    # Map our preprocessing function to every element in our dataset, taking
    # advantage of multithreading
    dataset = dataset.map(read_jpg, num_parallel_calls=threads)
    dataset = dataset.map(preproc_fn, num_parallel_calls=threads)
    # sloowww https://github.com/tensorflow/tensorflow/issues/14857
    #if shuffle:
    #    dataset = dataset.shuffle(num_x)
   
    # It's necessary to repeat our data for all epochs 
    dataset = dataset.repeat().batch(batch_size)
    return dataset
