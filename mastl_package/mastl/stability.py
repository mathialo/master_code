from mastl.models.deepMRInet import dc_layer, cnn_layer
from mastl.models.common import inner_fft_4d, inner_ifft_4d, convert_complex_format, non_gathering_boolean_mask
from mastl.training import optimistic_restore

import numpy as np
import tensorflow as tf
import sigpy.imaging as img


def deepMRInet(model_input, perturbation, sampling_mask):
    concictency_pars = [0.2, 0.2, 0.2, 0.2]
    filternums = [32, 32, 32, 32]
    filtersizes = [5, 5, 3, 3]
    datatype = tf.complex64


    def model(tensor_in, is_training):
        last_layer = inner_ifft_4d(tensor_in)

        for i in range(4):
            last_layer = convert_complex_format(last_layer)
            cnn_result = cnn_layer(last_layer, filtersizes, filternums,
                                   is_training,
                                   use_batchnorm=False,
                                   activation=tf.nn.leaky_relu,
                                   last_activation=tf.nn.tanh,
                                   initialization=lambda: None)

            cnn_result = convert_complex_format(cnn_result)
            last_layer = dc_layer(cnn_result, tensor_in, sampling_mask,
                                  concictency_pars[i])

        return last_layer

    def tf_sampling_op(signal):
        kspace = inner_fft_4d(tf.cast(signal, tf.complex64))
        return non_gathering_boolean_mask(kspace, sampling_mask)


    # Configure model_input/output
    perturbed_input = tf_sampling_op(model_input + tf.cast(perturbation, datatype))

    return model(perturbed_input, False)


def create_deepMRInet_stuff():
    batch_size = 1
    x_size = 368
    y_size = 368
    channels_num = 1
    datatype = tf.complex64

    model_input = tf.placeholder(datatype,
                                 shape=[batch_size, y_size, x_size, channels_num])

    perturbation = tf.get_variable("delta",
                                   shape=[batch_size, y_size, x_size, channels_num],
                                   dtype=tf.float32)

    return model_input, perturbation


def run_tests(testnums,
              session,
              model_input,
              perturbation,
              model_output,
              vanilla_placement,
              parseval_placement,
              perturbation_scale=1):

    # Parseval tests
    optimistic_restore(session, parseval_placement)

    for testnum in testnums:
        img.display(
            np.abs(
                np.squeeze(
                    session.run(
                        model_output,
                        feed_dict={model_input: np.load("{}_original.npy".format(testnum)),
                                   perturbation: np.zeros_like(np.load("{}_perturbation_final.npy".format(testnum)))
                        }
                    )
                )
            ),
            filename="{}_recon_parseval.png".format(testnum),
            show=False
        )
        img.display(
            np.abs(
                np.squeeze(
                    session.run(
                        model_output,
                        feed_dict={model_input: np.load("{}_original.npy".format(testnum)),
                                   perturbation: perturbation_scale*np.load("{}_perturbation_final.npy".format(testnum))
                        }
                    )
                )
            ),
            filename="{}_recon_parseval_perturbed.png".format(testnum),
            show=False
        )

    # Vanilla tests
    optimistic_restore(session, vanilla_placement)

    for testnum in testnums:
        img.display(
            np.abs(
                np.squeeze(
                    session.run(
                        model_output,
                        feed_dict={model_input: np.load("{}_original.npy".format(testnum)),
                                   perturbation: np.zeros_like(np.load("{}_perturbation_final.npy".format(testnum)))
                        }
                    )
                )
            ),
            filename="{}_recon_vanilla.png".format(testnum),
            show=False
        )
        img.display(
            np.abs(
                np.squeeze(
                    session.run(
                        model_output,
                        feed_dict={model_input: np.load("{}_original.npy".format(testnum)),
                                   perturbation: perturbation_scale*np.load("{}_perturbation_final.npy".format(testnum))
                        }
                    )
                )
            ),
            filename="{}_recon_vanilla_perturbed.png".format(testnum),
            show=False
        )

    # Save original and perturbed
    for testnum in testnums:
        img.display(
            np.abs(
                np.squeeze(
                    np.load("{}_original.npy".format(testnum))
                )
            ),
            filename="{}_original.png".format(testnum),
            show=False
        )
        img.display(
            np.abs(
                np.squeeze(
                    np.load("{}_original.npy".format(testnum)) + perturbation_scale*np.load("{}_perturbation_final.npy".format(testnum))
                )
            ),
            filename="{}_perturbed.png".format(testnum),
            show=False
        )