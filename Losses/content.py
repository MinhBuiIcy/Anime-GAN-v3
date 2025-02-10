import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.models import Model

from losses.vgg19 import build_vgg19
from tensorflow.keras.applications.vgg19 import preprocess_input


def content_loss(original_image, guided_image, model, layers=[15]):
    model = build_vgg19(model, layers)

    norm_original_image = (original_image + 1) * 127.5
    norm_guided_image = (guided_image + 1) * 127.5

    norm_original_image = preprocess_input(norm_original_image)
    norm_guided_image = preprocess_input(norm_guided_image)

    original_features = model(norm_original_image)
    guided_features = model(norm_guided_image)

    loss = tf.reduce_mean(tf.abs(original_features - guided_features))

    return loss