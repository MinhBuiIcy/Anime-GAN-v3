import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.models import Model

from tensorflow.keras.applications.vgg19 import preprocess_input
from losses.vgg19 import build_vgg19
from support_fun import tensorflow_felzenszwalb_superpixel

def region_smoothing_loss(original_image, guided_image, model, layers=[15]):
    model = build_vgg19(model, layers)

    sp_original_image = tensorflow_felzenszwalb_superpixel(original_image)
    sp_guided_image = tensorflow_felzenszwalb_superpixel(guided_image)

    sp_original_image = (sp_original_image + 1) * 127.5
    sp_guided_image = (sp_guided_image + 1) * 127.5
    norm_guided_image = (guided_image + 1) * 127.5

    sp_original_image = preprocess_input(sp_original_image)
    sp_guided_image = preprocess_input(sp_guided_image)
    norm_guided_image = preprocess_input(norm_guided_image)

    sp_original_features = model(sp_original_image)
    sp_guided_features = model(sp_guided_image)
    guided_features = model(norm_guided_image)

    ori_loss = tf.reduce_mean(tf.abs(sp_original_features - guided_features))
    guided_loss = tf.reduce_mean(tf.abs(sp_guided_features - guided_features))
    
    loss = 0.2 * ori_loss + guided_loss
    return loss


