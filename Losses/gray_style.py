import tensorflow as tf

from tensorflow.keras.applications.vgg19 import preprocess_input
from losses.vgg19 import build_vgg19
from support_fun import gram_matrix

def gray_style_loss(anime_image, guided_image, model, layers=[5, 9, 15], coffs=[0.1, 5.0, 25.0]):
    model = build_vgg19(model, layers)

    norm_anime_image = (anime_image + 1) * 127.5
    norm_guided_image = (guided_image + 1) * 127.5

    norm_anime_image = preprocess_input(norm_anime_image)
    norm_guided_image = preprocess_input(norm_guided_image)

    anime_features = model(norm_anime_image)
    guided_features = model(norm_guided_image)

    total_loss = 0

    for i in range(3):
        anime_tensor = anime_features[i]         # Shape: (batch_size, height, width, channels)
        guided_tensor = guided_features[i] # Shape: (batch_size, height, width, channels)

        # Compute the Gram matrices for the batch
        gram_anime = gram_matrix(anime_tensor)         # Shape: (batch_size, channels, channels)
        gram_guided = gram_matrix(guided_tensor) # Shape: (batch_size, channels, channels)

        # Get the normalization factors
        batch_size, height, width, num_channels = tf.shape(anime_tensor)[0], tf.shape(anime_tensor)[1], tf.shape(anime_tensor)[2], tf.shape(anime_tensor)[3]
        spatial_elements = tf.cast(height * width, tf.float32)  # Total spatial elements

        # Compute style loss for each batch element
        layer_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(gram_anime - gram_guided), axis=[1, 2]) /
            (4 * (tf.cast(num_channels, tf.float32) ** 2) * (spatial_elements ** 2))
        )

        # Accumulate loss for the current layer
        total_loss += layer_loss * coffs[i]

    return total_loss / 3