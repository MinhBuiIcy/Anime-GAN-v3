import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,
    BatchNormalization, Activation, Dense, Dropout, Flatten, Multiply, Add, Lambda, SpatialDropout2D, Reshape, GlobalMaxPooling2D, Concatenate, Layer,
    LeakyReLU
)
from tensorflow.keras.initializers import HeNormal, GlorotNormal
from tensorflow.keras.regularizers import l2
import support_fun
from model.layers import InstanceNormalization, SpatialAttentionLayer
from skimage.segmentation import felzenszwalb



def cbam_block(input_tensor, reduction_ratio=4):
    """CBAM block for feature refinement."""
    # Channel Attention
    channel_avg_pool = GlobalAveragePooling2D()(input_tensor)
    channel_max_pool = GlobalMaxPooling2D()(input_tensor)

    channel_avg_pool = Dense(input_tensor.shape[-1] // reduction_ratio, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-3))(channel_avg_pool)
    channel_avg_pool = Dense(input_tensor.shape[-1], activation='sigmoid', kernel_initializer=GlorotNormal(), kernel_regularizer=l2(1e-3))(channel_avg_pool)

    channel_max_pool = Dense(input_tensor.shape[-1] // reduction_ratio, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-3))(channel_max_pool)
    channel_max_pool = Dense(input_tensor.shape[-1], activation='sigmoid', kernel_initializer=GlorotNormal(), kernel_regularizer=l2(1e-3))(channel_max_pool)

    channel_avg_pool = Reshape((1, 1, input_tensor.shape[-1]))(channel_avg_pool)
    channel_max_pool = Reshape((1, 1, input_tensor.shape[-1]))(channel_max_pool)

    channel_attention = Add()([channel_avg_pool, channel_max_pool])
    channel_attention = Multiply()([input_tensor, channel_attention])

    # Spatial Attention
    spatial_attention = SpatialAttentionLayer()(channel_attention)

    spatial_attention = Conv2D(1, (7, 7), activation='sigmoid', padding='same')(spatial_attention)
    spatial_attention = Multiply()([channel_attention, spatial_attention])

    spatial_attention = InstanceNormalization()(spatial_attention)

    return spatial_attention

# Define the functions (fine_grained_revision and guided_filter_tf)
@tf.function
def fine_grained_revision(ori_image):
    image = (ori_image + 1) / 2
    gray = tf.image.rgb_to_grayscale(image)
    
    # Laplacian filter for edge detection
    kernel = tf.constant([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])  # Shape for conv2d
    edges = tf.nn.conv2d(gray[tf.newaxis, ..., tf.newaxis], kernel, strides=[1, 1, 1, 1], padding="SAME")[0]
    
    edges = edges - tf.reduce_mean(edges)

    edges = tf.clip_by_value(edges * 0.15, 0.0, 1.0)
    enhanced = tf.add(image, edges[..., 0])  # Broadcast edges over RGB channels
    enhanced = tf.clip_by_value(enhanced, 0.0, 1.0)
    
    return enhanced * 2 - 1

@tf.function
def guided_filter_tf(ori_image, radius=7, eps=1e-2):
    image = (ori_image + 1) / 2
    """Applies an edge-preserving smoothing filter using TensorFlow operations."""
    guidance = tf.image.rgb_to_grayscale(image)  # (B, H, W, 1)
    # Normalize guidance to have same mean as original image
    #guidance = guidance - tf.reduce_mean(guidance, axis=[1, 2], keepdims=True) + tf.reduce_mean(image, axis=[1, 2], keepdims=True)
    guidance = guidance - tf.reduce_mean(guidance) + tf.reduce_mean(image)
    kernel_size = 2 * radius + 1
    kernel = tf.ones((kernel_size, kernel_size, 1, 1), dtype=tf.float32) / (kernel_size ** 2)
    smoothed = tf.nn.depthwise_conv2d(guidance, kernel, strides=[1, 1, 1, 1], padding='SAME')
    output = (image * (1 - eps)) + (smoothed * eps)
    return (output * 2) - 1

def tensorflow_felzenszwalb_superpixel(image_tensor):
    """
    Applies the Felzenszwalb Superpixel segmentation to a TensorFlow image tensor.
    """
    def felzenszwalb_opencv(image):
        """
    Applies Felzenszwalb segmentation using OpenCV & Scikit-Image on a batch of images.
        """
        image_np = image.numpy()  # Convert Tensor to NumPy array

    # Check if the input is a batch (B, H, W, C) or a single image (H, W, C)
        if image_np.ndim == 3:  # Single image case, add batch dimension
            image_np = np.expand_dims(image_np, axis=0)  # Shape: (1, H, W, C)

        batch_size, height, width, channels = image_np.shape
        processed_images = np.zeros_like(image_np, dtype=np.float32)  # Output container

        for i in range(batch_size):  # Loop through batch
            img = image_np[i]

        # Convert to uint8 if needed
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

        # Ensure image is RGB
            if img.shape[-1] == 1:  # If grayscale, convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Convert to LAB color space
            lab_image = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

        # Apply Felzenszwalb segmentation
            segments = felzenszwalb(lab_image, scale=100, sigma=0.5, min_size=50)

        # Ensure shape matches input image (H x W)
            if segments.shape[:2] != img.shape[:2]:
                raise ValueError(f"Segment shape {segments.shape} does not match image shape {img.shape}")

        # Create an edge mask for visualization
            edges = cv2.Laplacian(segments.astype(np.uint8), cv2.CV_8U, ksize=3)
            img[edges > 0] = [0, 0, 255]  # Draw edges in red

        # Normalize back to [0,1]
            processed_images[i] = img.astype(np.float32) / 255.0  

        return processed_images  # Return batch of processed images

    output = tf.py_function(func=felzenszwalb_opencv, inp=[image_tensor], Tout=tf.float32)
    output = tf.ensure_shape(output, image_tensor.shape)

    return output

def rgb2gray(rgb):
    """
    Convert an RGB image to grayscale.
    Args:
        rgb: Tensor of shape [batch, height, width, channels] with values in range [0, 1].
    Returns:
        Grayscale image tensor of shape [batch, height, width, 1].
    """
    # Rescale [-1,1] to [0,1]
    rgb = (rgb + 1) / 2.0  
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)

    coeffs = tf.constant([0.299, 0.587, 0.114], dtype=tf.float32)
    gray = tf.reduce_sum(rgb * coeffs, axis=-1, keepdims=True)

    # Normalize back to [-1,1]
    gray = (gray * 2.0) - 1.0  
    return gray

def rgb2lab(rgb):
    """
    Convert an RGB image to LAB color space.
    Args:
        rgb: Tensor of shape [batch, height, width, 3] with values in range [0, 1].
    Returns:
        LAB image tensor of shape [batch, height, width, 3].
    """
    # Define transformation matrix
    matrix = tf.constant([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=tf.float32)
    rgb = (rgb + 1) / 2
    # Normalize RGB to [0, 1]
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)

    # Apply gamma correction
    mask = rgb > 0.04045
    rgb = tf.where(mask, tf.pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)

    # Convert RGB to XYZ while maintaining batch, height, and width
    shape = tf.shape(rgb)  # Store original shape
    rgb = tf.reshape(rgb, [-1, 3])  # Flatten only for matrix multiplication
    xyz = tf.matmul(rgb, tf.transpose(matrix))
    xyz = tf.reshape(xyz, shape)  # Restore shape [batch, height, width, 3]

    # Normalize XYZ for D65
    xyz_ref = tf.constant([0.95047, 1.00000, 1.08883], dtype=tf.float32)
    xyz /= xyz_ref

    # Convert XYZ to LAB
    epsilon = 0.008856
    kappa = 903.3
    f_xyz = tf.where(xyz > epsilon, tf.pow(xyz, 1/3), (kappa * xyz + 16) / 116)

    # Extract L, a, b while preserving spatial structure
    L = (116 * f_xyz[..., 1]) - 16  # Shape: [batch, height, width]
    a = 500 * (f_xyz[..., 0] - f_xyz[..., 1])  # Shape: [batch, height, width]
    b = 200 * (f_xyz[..., 1] - f_xyz[..., 2])  # Shape: [batch, height, width]

    L = (L / 50.0) - 1
    a = a / 128.0
    b = b / 128.0
    # Stack into LAB image
    lab = tf.stack([L, a, b], axis=-1)  # Correctly shaped [batch, height, width, 3]

    return lab

def gram_matrix(X):
    # Get the batch size, spatial dimensions, and number of channels
    batch_size, height, width, num_channels = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2], tf.shape(X)[3]

    # Reshape each feature map to (batch_size, height * width, channels)
    X = tf.reshape(X, (batch_size, height * width, num_channels))

    # Compute the Gram matrix for each batch element
    # (batch_size, channels, channels)
    gram_matrices = tf.matmul(X, X, transpose_a=True) / tf.cast(height * width, tf.float32)

    return gram_matrices