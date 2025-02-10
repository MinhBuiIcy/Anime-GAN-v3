import tensorflow as tf

def m_adv_loss(main_gan, original_image):
    output = main_gan(original_image, training=True)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(
        tf.ones_like(output), output
    )

    return loss