import functools
import tensorflow as tf
import tensorflow.contrib as tfcontrib
import functools
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
import numpy as np
import os

def conv_block(input_tensor, num_filters):
    # capa convolucional con kernel 3x3
    # https://www.youtube.com/watch?v=LgFNRIFxuUo
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    # batch-normalization truco para acelerar entrenamiento forzando normalización de cada batch (en cada paso de entrenamiento)
    # https://arxiv.org/pdf/1502.03167.pdf
    encoder = layers.BatchNormalization()(encoder)
    # funciones de activación: sigmoide, lineal, relu, leaky-relu, swish
    # https://www.learnopencv.com/understanding-activation-functions-in-deep-learning/
    encoder = layers.Activation('relu')(encoder)
    # 
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

# combine both dice loss and binary cross entropy
def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss