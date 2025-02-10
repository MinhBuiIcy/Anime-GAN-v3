import tensorflow as tf

from support_fun import fine_grained_revision

def dm_loss(discriminator, main_image, revised_image):
    revised_output = discriminator(revised_image)
    main_output = discriminator(main_image)

    main_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(
        tf.fill(tf.shape(main_output), 0.1), main_output
    )
    support_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(
        tf.fill(tf.shape(revised_output), 0.9), revised_output
    )
    total_loss = main_loss + support_loss
    return total_loss