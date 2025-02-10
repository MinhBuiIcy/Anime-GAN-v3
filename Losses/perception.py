import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.models import Model

from losses.vgg19 import build_vgg19
from tensorflow.keras.applications.vgg19 import preprocess_input

def perception_loss(revised_image, main_image, model, layers=[15]):
    model = build_vgg19(model, layers)

    norm_revised_image = (revised_image + 1) * 127.5
    norm_main_image = (main_image + 1) * 127.5

    norm_revised_image = preprocess_input(norm_revised_image)
    norm_main_image = preprocess_input(norm_main_image)

    revised_features = model(norm_revised_image)
    main_features = model(norm_main_image)

    loss = tf.reduce_mean(tf.abs(revised_features - main_features))

    return loss


