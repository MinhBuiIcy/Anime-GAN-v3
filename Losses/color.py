import tensorflow as tf

from support_fun import rgb2lab

def color_loss(original_image, guided_image):
    """
    Compute color loss between two images in LAB color space.
    
    Args:
        original_image: Tensor of shape [batch, height, width, 3], RGB values in range [0, 1].
        guided_image: Tensor of shape [batch, height, width, 3], RGB values in range [0, 1].
        
    Returns:
        Scalar Tensor representing the mean squared color loss.
    """
    # Convert both images to LAB space
    original_lab = rgb2lab(original_image)  # Shape: [batch, height, width, 3]
    guided_lab = rgb2lab(guided_image)  # Shape: [batch, height, width, 3]

    # Compute Mean Squared Error (MSE) in LAB space
    loss = tf.reduce_mean(tf.abs(original_lab - guided_lab))  # MSE loss

    return loss  # Scalar