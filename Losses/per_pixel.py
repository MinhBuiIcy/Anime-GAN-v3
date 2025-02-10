import tensorflow as tf

def per_pixel_loss(revised_image, main_image, loss_type='L1'):
    """
    Computes the final per-pixel loss between real and generated images.

    Args:
        real_image (tf.Tensor): Tensor of shape (batch, height, width, channels).
        generated_image (tf.Tensor): Tensor of shape (batch, height, width, channels).
        loss_type (str): Type of loss to compute ('L1' for MAE, 'L2' for MSE).

    Returns:
        tf.Tensor: Scalar loss value.
    """
    if loss_type == 'L1':  # Mean Absolute Error (MAE)
        loss = tf.reduce_mean(tf.abs(revised_image - main_image))
    elif loss_type == 'L2':  # Mean Squared Error (MSE)
        loss = tf.reduce_mean(tf.square(revised_image - main_image))
    else:
        raise ValueError("Invalid loss type. Use 'L1' or 'L2'.")

    return loss  # A single scalar value
