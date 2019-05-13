import tensorflow as tf
import numpy as np

def non_gathering_boolean_mask(x, mask):
    mask = np.repeat(mask[np.newaxis, :, :], int(x.shape[0]), axis=0)
    mask = np.repeat(mask[:, :, :, np.newaxis], int(x.shape[3]), axis=3)

    zeros = tf.zeros_like(x, dtype=x.dtype)

    return tf.where(mask, x, zeros)


def inner_fft_4d(tensor):
    pre_transpose = tf.transpose(tensor, [0, 3, 1, 2])
    dft_coeffs = tf.spectral.fft2d(pre_transpose)
    return tf.transpose(dft_coeffs, [0, 2, 3, 1])


def inner_ifft_4d(tensor):
    pre_transpose = tf.transpose(tensor, [0, 3, 1, 2])
    dft_coeffs = tf.spectral.ifft2d(pre_transpose)
    return tf.transpose(dft_coeffs, [0, 2, 3, 1])


def convert_complex_format(tensor_in):
    """
    Converts complex number format from using the actual tf.complex64 type (used
    for tf.spectral.fft) and having the real and imaginary parts as separate
    channels_num in the tensor (used for everything else).

    Args:
        tensor_in:      A 4D tensor [batch, height, width, channels_num] where either:

                          * The channels_num dimension is 1, and the type is complex

                                or

                          * The channels_num dimemsion is 2, and the type is some
                            type of real number (float, int)

    Returns:
        The tensor where the complex type has been toggled.
    """
    if tensor_in.dtype.is_complex:
        # Convert to 2 channels_num
        return tf.concat([tf.real(tensor_in), tf.imag(tensor_in)], axis=3)

    else:
        # Convert to complex type
        return tf.complex(real=tensor_in[:, :, :, 0:1], imag=tensor_in[:, :, :, 1:])

