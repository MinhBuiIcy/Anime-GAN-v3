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
from model.layers import LADELayer, ReflectionPadding2D

def build_main_gan(generator, discriminator):
    inputs = Input(shape=(256, 256, 3))

    gen_outputs = generator(inputs)
    dis_outputs = discriminator(gen_outputs)

    return Model(inputs, dis_outputs, name="main_gan")