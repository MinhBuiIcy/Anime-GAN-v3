import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,
    BatchNormalization, Activation, Dense, Dropout, Flatten, Multiply, Add, Lambda, SpatialDropout2D, Reshape, GlobalMaxPooling2D, Layer, UpSampling2D,
    LeakyReLU
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal, GlorotNormal
from tensorflow.keras.regularizers import l2

import support_fun
from model.layers import InstanceNormalization, LADELayer, ReflectionPadding2D, ResidualBlock

def build_support_generator(encoder):
    inputs = Input(shape=(256, 256, 3))  # Adjust input shape if necessary
    
    skip1, skip2, skip3, encoded_outputs = encoder(inputs)  # Use shared encoder
    # Attention block
    x = support_fun.cbam_block(encoded_outputs)
    x = Add()([x, 0.2 * encoded_outputs])
    x = LeakyReLU(negative_slope=0.2)(x)

    # First upsample block
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)

    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = Conv2D(128, kernel_size=3, strides=1, padding="valid", kernel_initializer=HeNormal())(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = Conv2D(128, kernel_size=3, strides=1, padding="valid", kernel_initializer=HeNormal())(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Add()([x, 0.2 * skip3])
    # Second upsample block
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)

    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding="valid", kernel_initializer=HeNormal())(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding="valid", kernel_initializer=HeNormal())(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Add()([x, 0.2 * skip2])

    # Third upsample block
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)

    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = Conv2D(132, kernel_size=3, strides=1, padding="valid", kernel_initializer=HeNormal())(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = Conv2D(32, kernel_size=3, strides=1, padding="valid", kernel_initializer=HeNormal())(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Add()([x, 0.2 * skip1])

    # Output Layer
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = Conv2D(3, kernel_size=7, strides=1, padding="valid", kernel_initializer=HeNormal())(x)
    x = InstanceNormalization()(x)
    outputs = Activation("tanh")(x)


    return Model(inputs, outputs, name="support_generator")
