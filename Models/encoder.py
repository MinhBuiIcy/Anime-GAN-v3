import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,
    BatchNormalization, Activation, Dense, Dropout, Flatten, Multiply, Add, Lambda, SpatialDropout2D, Reshape, GlobalMaxPooling2D, Layer, UpSampling2D,
    LeakyReLU
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal, GlorotNormal
from tensorflow.keras.regularizers import l2

from model.layers import InstanceNormalization, LADELayer, ReflectionPadding2D


def build_encoder(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    # First downsample block
    x = ReflectionPadding2D(padding=(3, 3))(inputs)
    x = Conv2D(32, kernel_size=7, strides=1, padding="valid", kernel_initializer=HeNormal())(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    skip1 = x

    # Second downsample block
    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = Conv2D(32, kernel_size=3, strides=2, padding="valid", kernel_initializer=HeNormal())(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding="valid", kernel_initializer=HeNormal())(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    skip2 = x

    # Third downsample block
    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = Conv2D(64, kernel_size=3, strides=2, padding="valid", kernel_initializer=HeNormal())(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = Conv2D(128, kernel_size=3, strides=1, padding="valid", kernel_initializer=HeNormal())(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    skip3 = x

    # Fourth downsample block
    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = Conv2D(128, kernel_size=3, strides=2, padding="valid", kernel_initializer=HeNormal())(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = Conv2D(128, kernel_size=3, strides=1, padding="valid", kernel_initializer=HeNormal())(x)
    x = InstanceNormalization()(x)
    outputs = LeakyReLU(negative_slope=0.2)(x)

    return Model(inputs, [skip1, skip2, skip3, outputs], name="encoder")