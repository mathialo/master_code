from mastl.models.deepMRInet import dc_layer, cnn_layer
from mastl.models.common import inner_fft_4d, inner_ifft_4d, convert_complex_format, non_gathering_boolean_mask
from mastl.data import DataSet
import mastl._iohelp as io
import mastl.parsnet.constraints
import mastl.losses

import tensorflow as tf
import numpy as np
import matplotlib as plt
import hdf5storage


num_tests = 5


# Hyperparameters
ridgereg_par = 1
learning_rate = 0.001
momentum = 0.9
initial_size = 1e-5
steps = 50


io.log("Lambda: %g,  Gamma: %g,  Eta: %g,  Tau: %g" % (ridgereg_par, momentum, learning_rate, initial_size))
io.log("Loading dataset")

sampling_mask = np.fft.fftshift(np.load("gaussian_pattern_25.npy"))
dataset = DataSet(
    "/mn/kadingir/vegardantun_000000/nobackup/GLOBUS/knee/coronal_pd",
    r"espirit[23]\d\.mat",
    hdf5storage.loadmat,
    data_key="reference",
    test_rate=0,
    scale=True,
    crop=(368, 368),
    cache=True,
    augment=False
)

io.log("Configuring model")

# Configure model in the same manner as deepMRInet.py
concictency_pars = [0.2, 0.2, 0.2, 0.2]
filternums = [32, 32, 32, 32]
filtersizes = [5, 5, 3, 3]
batch_size = 1
x_size = 368
y_size = 368
channels_num = 1
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
        last_layer = dc_layer(cnn_result, tensor_in, sampling_mask, concictency_pars[i])

    return last_layer


def sample_op(input):
    fft_input = np.fft.fft2(input)
    fft_input[np.invert(sampling_mask)] = 0
    return fft_input


def tf_sampling_op(signal):
    kspace = inner_fft_4d(tf.cast(signal, tf.complex64))
    return non_gathering_boolean_mask(kspace, sampling_mask)


# Configure model_input/output
is_training = tf.placeholder(tf.bool)
model_input = tf.placeholder(datatype, shape=[batch_size, y_size, x_size, channels_num])
true_output = tf.placeholder(datatype, shape=[batch_size, y_size, x_size, channels_num])

perturbation = tf.get_variable("delta", shape=[batch_size, y_size, x_size, channels_num], dtype=tf.float32, initializer=tf.initializers.random_uniform(0, initial_size))
perturbed_input = tf_sampling_op(model_input + tf.cast(perturbation, datatype))

model_output = model(perturbed_input, is_training)
model_loss = mastl.losses.l2(perturbed_input, true_output)


# Define Q(r)
precalculated_output = tf.placeholder(tf.complex64, shape=[batch_size, y_size, x_size, channels_num])
recon_error = tf.abs(tf.norm(model_output - precalculated_output))
perturbation_norm = tf.norm(perturbation)
q_function = 0.5*tf.square(recon_error) - 0.5*ridgereg_par*tf.square(perturbation_norm)

# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
opt_step = optimizer.minimize(-q_function, var_list=[perturbation])


# Load trained model
def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
    """
    By StackOverflow user Lior at
    https://stackoverflow.com/questions/47997203/tensorflow-restore-if-present


    Args:
        session:        session object
        save_file:      save file to load
        graph:          graph to act upon
    """
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    for var_name, saved_var_name in var_names:
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(session, save_file)

io.log("Loading pre-trained weights")
sess = tf.Session()

# Init everything, and load those who can be loaded from checkpoint (new variables have been made, thus everything cannot be loaded)
sess.run(tf.global_variables_initializer())
optimistic_restore(sess, "trained_vanilla/last_run.ckpt")


# Run tests with the same perturbs as the ones found for vanilla
for i in range(2, 6):
    print("Repeating recon for test {}".format(i))
    image = np.load("originals_perturbs/{}_original.npy".format(i))
    pert = np.load("originals_perturbs/{}_perturbation_final.npy".format(i))

    recon_original = sess.run(model_output, feed_dict = {
        model_input: image,
        perturbation: np.zeros_like(pert),
        is_training: False
    })
    recon_pert = sess.run(model_output, feed_dict = {
        model_input: image,
        perturbation: pert,
        is_training: False
    })

    np.save("results_vanilla/{}_reconstruction_original.npy".format(i), recon_original)
    np.save("results_vanilla/{}_reconstruction_perturbation.npy".format(i), recon_pert)
    # np.save("originals_perturbs/{}_perturbed.npy".format(i), image + pert)



sess.close()
