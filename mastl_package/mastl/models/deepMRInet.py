import tensorflow as tf
import numpy as np

from .common import convert_complex_format, non_gathering_boolean_mask, inner_fft_4d, \
    inner_ifft_4d


def cnn_layer(tensor_in,
              filter_sizes,
              filter_nums,
              training,
              use_batchnorm=True,
              activation=tf.nn.relu,
              last_activation=tf.nn.relu,
              initialization=tf.keras.initializers.he_normal,
              constraint=None,
              scale_filter=False):
    """
    A simple CNN implemented after the template in [1].

    Args:
        tensor_in (TF tensor):              Input to CNN. A 4D tensor consisting of
                                            [batch, height, width, channels_num].
        filter_sizes (iterable of ints):    A list of filter sizes. The length of the list
                                            will determine the depth of the CNN.
        filter_nums (iterable of ints):     A list of filter numbers. The length must be
                                            the same as filter_sizes, and the last entry
                                            must be the same at the number of channels_num in
                                            tensor_in.
        training (bool):                    Whether network is in training or inference mode
        use_batchnorm (bool):               Whether to use batch normalization or not
        activation (callable):              Activation function
        last_activation (callable):         Activation function of last layer (projection layer)
        initialization (callable):          Initialization scheme
        constraint (callable):              Kernel constraint
        scale_filter (boolean):             Scale layer with filter sizes

    Returns:
        TF tensor: Output tensor of the network. A 4D tensor consisting of [batch, height,
        width, channels_num].

    Raises:
        ValueError: If the specifications of the lists doesn't match up.

    """
    if len(filter_sizes) != len(filter_nums):
        raise ValueError(
            "filter_sizes has length {}, while filter_nums has length {}. ".format(
                len(filter_sizes), len(filter_nums)
            )
            + "These must be the same"
        )

    last_layer = tensor_in

    for filter_size, filter_num in zip(filter_sizes, filter_nums):
        last_layer = tf.layers.conv2d(
            last_layer,
            filter_num,
            (filter_size, filter_size),
            padding="SAME",
            use_bias=True,
            activation=activation,
            kernel_initializer=initialization(),
            trainable=True,
            kernel_constraint=constraint
        )

        if scale_filter:
            last_layer = last_layer / filter_size

        if use_batchnorm:
            last_layer = tf.layers.batch_normalization(
                last_layer,
                training=training
            )

    last_layer = tf.layers.conv2d(
        last_layer,
        2,
        (3, 3),
        padding="SAME",
        use_bias=True,
        activation=last_activation,
        kernel_initializer=initialization(),
        trainable=True,
        kernel_constraint=constraint
    )

    if scale_filter:
        last_layer = last_layer / filter_size
        convex_par = tf.sigmoid(tf.Variable(initial_value=0, trainable=True))
        return convex_par*last_layer + (1-convex_par)*tensor_in

    else:
        return last_layer + tensor_in



def dc_layer(tensor_in, original_samples, subsample_mask, consistency_par):
    """
    Data consistency layer as defined in [1]

    Args:
        tensor_in (TF tensor):          Input to layer. A 4D tensor consisting of [batch,
                                        height, width, channels_num]
        original_samples (TF tensor):   Original sampled coeffs, padded such that the dim
                                        is the same as tensor_in.
        subsample_mask (np.array):      A mask of the downsampling pattern. A boolean
                                        matrix where True means the sample was included.
        consistency_par (float):        Consistency parameter. What is refered to as
                                        lambda in [1].

    Returns:
        TF tensor: Output tensor of layer
    """
    if consistency_par == 0:
        return tensor_in

    reconstructed_dft_coeffs = inner_fft_4d(tensor_in)

    new_zks = non_gathering_boolean_mask(reconstructed_dft_coeffs, subsample_mask)
    new_zks = (new_zks + consistency_par * original_samples) / (1 + consistency_par)

    g = non_gathering_boolean_mask(reconstructed_dft_coeffs,
                                   np.invert(subsample_mask)) + new_zks

    return inner_ifft_4d(g)


def model(tensor_in, subsample_mask, num_layers, filter_sizes, filter_nums,
          consistency_par):
    last_layer = inner_ifft_4d(tensor_in)

    for i in range(num_layers):
        last_layer = convert_complex_format(last_layer)
        cnn_result = cnn_layer(last_layer, filter_sizes, filter_nums)

        cnn_result = convert_complex_format(cnn_result)
        last_layer = dc_layer(cnn_result, tensor_in, subsample_mask, consistency_par)

    return last_layer

# REFERENCES:
# [1] Jo Schlemper et al. "A deep cascade of concolutional neural networks for MR image
#     reconstruction". In: International Conference on Information Processing in Medical
#     Imaging. Springer. 2017.
