import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,
    BatchNormalization, Activation, Dense, Dropout, Flatten, Multiply, Add, Lambda, SpatialDropout2D, Reshape, GlobalMaxPooling2D, Concatenate, Layer
)
from tensorflow.keras.initializers import HeNormal, GlorotNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class LADELayer(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(LADELayer, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.conv = None  # Will be initialized in build

    def build(self, input_shape):
        C = input_shape[-1]
        self.conv = Conv2D(C, kernel_size=1, strides=1, padding="same", 
                           kernel_initializer=HeNormal(), use_bias=False)
        super(LADELayer, self).build(input_shape)

    def call(self, x):
        # Compute mean and std across spatial dimensions (H, W) per channel
        mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        std = tf.math.sqrt(tf.reduce_mean(tf.square(x - mean), axis=[1, 2], keepdims=True) + self.epsilon)

        # Normalize x
        x_norm = (x - mean) / (std + self.epsilon)

        # Apply 1x1 Pointwise Convolution to get P(x)
        px = self.conv(x)

        # Compute β and γ from P(x)
        beta = tf.reduce_mean(px, axis=[1, 2], keepdims=True)
        gamma = tf.math.sqrt(tf.reduce_mean(tf.square(px - beta), axis=[1, 2], keepdims=True) + self.epsilon)

        # Apply LADE transformation
        lade_output = gamma * x_norm + beta
        return lade_output

    def get_config(self):
        config = super(LADELayer, self).get_config()
        config.update({"epsilon": self.epsilon})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class SpatialAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttentionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        return tf.concat([avg_pool, max_pool], axis=-1)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
@register_keras_serializable()
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = padding

    def call(self, inputs):
        padding_width, padding_height = self.padding
        return tf.pad(inputs, [[0, 0], [padding_height, padding_height], [padding_width, padding_width], [0, 0]], 'REFLECT')

    def get_config(self):
        config = super(ReflectionPadding2D, self).get_config()
        config.update({"padding": self.padding})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable()
class RGBToGrayLayer(Layer):
    def __init__(self, **kwargs):
        super(RGBToGrayLayer, self).__init__(**kwargs)
        self.coeffs = tf.constant([0.299, 0.587, 0.114], dtype=tf.float32)
    
    def call(self, ori_inputs):
        inputs = (ori_inputs + 1) / 2
        output = tf.reduce_sum(inputs * self.coeffs, axis=-1, keepdims=True)
        return output * 2 - 1
    
    def get_config(self):
        config = super(RGBToGrayLayer, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
@register_keras_serializable()
class GuidedFilterLayer(tf.keras.layers.Layer):
    def __init__(self, radius=7, eps=1e-2, **kwargs):
        super(GuidedFilterLayer, self).__init__(**kwargs)
        self.radius = radius
        self.eps = eps
    
    def build(self, input_shape):
        kernel_size = 2 * self.radius + 1
        self.kernel = tf.ones((kernel_size, kernel_size, 1, 1), dtype=tf.float32) / (kernel_size ** 2)
        super(GuidedFilterLayer, self).build(input_shape)
    
    def call(self, ori_inputs):
        inputs = (ori_inputs + 1) / 2
        guidance = tf.image.rgb_to_grayscale(inputs)
        guidance = guidance - tf.reduce_mean(guidance) + tf.reduce_mean(inputs)
        smoothed = tf.nn.depthwise_conv2d(guidance, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        output = (inputs * (1 - self.eps)) + (smoothed * self.eps)
        return output * 2 - 1
    
    def get_config(self):
        config = super(GuidedFilterLayer, self).get_config()
        config.update({"radius": self.radius, "eps": self.eps})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
@register_keras_serializable()
class InstanceNormalization(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        # Learnable scaling and shifting parameters
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True,
            name="gamma"
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
            name="beta"
        )

    def call(self, inputs):
        # Calculate mean and variance across spatial dimensions (H, W)
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        # Normalize and apply learnable parameters
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({
            "epsilon": self.epsilon,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
@register_keras_serializable()
class ResidualBlock(Layer):
    """Custom Residual Block Layer"""
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.padding_layer = ReflectionPadding2D(padding=(1, 1))
        self.conv1 = Conv2D(filters, kernel_size=3, strides=1, padding="valid",
                            kernel_initializer=HeNormal())
        self.inst_norm1 = InstanceNormalization()
        self.relu1 = Activation("relu")
        self.conv2 = Conv2D(filters, kernel_size=3, strides=1, padding="valid",
                            kernel_initializer=HeNormal())
        self.inst_norm2 = InstanceNormalization()
        self.relu2 = Activation("relu")

    def call(self, inputs):
        x = self.padding_layer(inputs)
        x = self.conv1(x)
        x = self.inst_norm1(x)
        x = self.relu1(x)
        x = self.padding_layer(x)
        x = self.conv2(x)
        x = self.inst_norm2(x)
        x = self.relu2(x)
        return Add()([inputs, x])

    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({"filters": self.filters})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)