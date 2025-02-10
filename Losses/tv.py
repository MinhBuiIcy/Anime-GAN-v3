import tensorflow as tf


def total_loss(image_tensor):
    # Compute differences along the width and height dimensions
    width_diff = tf.abs(image_tensor[:, :-1, :-1, :] - image_tensor[:, 1:, :-1, :])
    height_diff = tf.abs(image_tensor[:, :-1, :-1, :] - image_tensor[:, :-1, 1:, :])

    # Compute total variation loss for the batch
    batch_total_loss = tf.reduce_sum(tf.pow(width_diff + height_diff, 1.25), axis=[1, 2, 3])

    # Sum the loss over the entire batch
    total_loss = tf.reduce_mean(batch_total_loss)

    return total_loss