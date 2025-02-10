import tensorflow as tf

def s_adv_loss(support_gan, original_image):
    output = support_gan(original_image, training=True)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(
        tf.ones_like(output), output
    )

    return loss