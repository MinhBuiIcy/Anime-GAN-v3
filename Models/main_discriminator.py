import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Activation, 
    LeakyReLU, SpectralNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal, GlorotNormal
from tensorflow.keras.regularizers import l2

from model.layers import InstanceNormalization, LADELayer, ReflectionPadding2D

def build_main_discriminator(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    # First downsample block
    x = ReflectionPadding2D(padding=(3, 3))(inputs)
    x = SpectralNormalization(Conv2D(32, kernel_size=7, strides=1, padding="valid", kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-3)))(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = SpectralNormalization(Conv2D(32, kernel_size=3, strides=2, padding="valid", kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-3)))(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = SpectralNormalization(Conv2D(64, kernel_size=3, strides=1, padding="valid", kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-3)))(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = SpectralNormalization(Conv2D(64, kernel_size=3, strides=2, padding="valid", kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-3)))(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = SpectralNormalization(Conv2D(128, kernel_size=3, strides=1, padding="valid", kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-3)))(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = SpectralNormalization(Conv2D(128, kernel_size=3, strides=2, padding="valid", kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-3)))(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = SpectralNormalization(Conv2D(256, kernel_size=3, strides=1, padding="valid", kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-3)))(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = SpectralNormalization(Conv2D(1, kernel_size=3, strides=1, padding="valid", kernel_initializer=HeNormal()))(x)
    x = Activation('sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x, name="main_discriminator")
    return model


