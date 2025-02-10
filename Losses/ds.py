import tensorflow as tf

from support_fun import fine_grained_revision, guided_filter_tf, rgb2gray

def ds_loss_ori(discriminator, guided_image, anime_image, blurred_anime_image):
    gray_support_output = rgb2gray(guided_image)
    gray_anime_image = rgb2gray(anime_image)
    gray_blurred_anime_image = rgb2gray(blurred_anime_image)

    gray_support_output = discriminator(gray_support_output)
    gray_anime_image = discriminator(gray_anime_image)
    gray_blurred_anime_image = discriminator(gray_blurred_anime_image)
    
    gf_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(gray_support_output), gray_support_output
    )
    a_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(gray_anime_image), gray_anime_image
    )
    e_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(gray_blurred_anime_image), gray_blurred_anime_image
    )
    total_loss = gf_loss + 0.5 * a_loss + e_loss
    return total_loss

def ds_loss(discriminator, image, type):
    gray_image = rgb2gray(image)

    gray_image_output = discriminator(gray_image)

    loss = 0

    if type == "a":
        loss = 0.5 * tf.keras.losses.BinaryCrossentropy(from_logits=False)(
            tf.fill(tf.shape(gray_image_output), 0.9), gray_image_output
        )
    else:
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(
            tf.fill(tf.shape(gray_image_output), 0.1), gray_image_output
        )
    
    return loss