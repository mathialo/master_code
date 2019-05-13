import tensorflow as tf
import numpy as np


def model(tensor_in, filtersizes, constraint=None, initializer=None):
    # Downconv
    layer1 = tf.layers.conv2d(
        tensor_in,
        filtersizes[0],
        kernel_size=(4, 4),
        strides=(1, 1),
        padding="SAME",
        activation=tf.nn.leaky_relu,
        kernel_initializer=initializer,
        kernel_constraint=constraint
    )
    layer2 = tf.layers.conv2d(
        layer1,
        filtersizes[1],
        kernel_size=(4, 4),
        strides=(2, 2),
        padding="SAME",
        activation=tf.nn.leaky_relu,
        kernel_initializer=initializer,
        kernel_constraint=constraint
    )
    layer3 = tf.layers.conv2d(
        layer2,
        filtersizes[2],
        kernel_size=(4, 4),
        strides=(2, 2),
        padding="SAME",
        activation=tf.nn.leaky_relu,
        kernel_initializer=initializer,
        kernel_constraint=constraint
    )
    layer4 = tf.layers.conv2d(
        layer3,
        filtersizes[3],
        kernel_size=(4, 4),
        strides=(2, 2),
        padding="SAME",
        activation=tf.nn.leaky_relu,
        kernel_initializer=initializer,
        kernel_constraint=constraint
    )
    layer5 = tf.layers.conv2d(
        layer4,
        filtersizes[4],
        kernel_size=(4, 4),
        strides=(2, 2),
        padding="SAME",
        activation=tf.nn.leaky_relu,
        kernel_initializer=initializer,
        kernel_constraint=constraint
    )
    layer6 = tf.layers.conv2d(
        layer5,
        filtersizes[5],
        kernel_size=(4, 4),
        strides=(2, 2),
        padding="SAME",
        activation=tf.nn.leaky_relu,
        kernel_initializer=initializer,
        kernel_constraint=constraint
    )

    # Upconv
    layer7 = tf.concat([
        tf.layers.conv2d_transpose(
            layer6,
            filtersizes[6],
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            activation=tf.nn.relu,
            kernel_initializer=initializer,
            kernel_constraint=constraint
        ),
        layer5],
        axis=-1
    )
    layer8 = tf.concat([
        tf.layers.conv2d_transpose(
            layer7,
            filtersizes[7],
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            activation=tf.nn.relu,
            kernel_initializer=initializer,
            kernel_constraint=constraint
        ),
        layer4],
        axis=-1
    )
    layer9 = tf.concat([
        tf.layers.conv2d_transpose(
            layer8,
            filtersizes[8],
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            activation=tf.nn.relu,
            kernel_initializer=initializer,
            kernel_constraint=constraint
        ),
        layer3],
        axis=-1
    )
    layer10 = tf.concat([
        tf.layers.conv2d_transpose(
            layer9,
            filtersizes[9],
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            activation=tf.nn.relu,
            kernel_initializer=initializer,
            kernel_constraint=constraint
        ),
        layer2],
        axis=-1
    )
    layer11 = tf.layers.conv2d_transpose(
        layer10,
        filtersizes[10],
        kernel_size=(4, 4),
        strides=(2, 2),
        padding="SAME",
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        kernel_constraint=constraint
    )
    projectionlayer = tf.layers.conv2d(
        layer11,
        int(tensor_in.shape[-1]),
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="SAME",
        activation=tf.nn.tanh,
        kernel_initializer=initializer,
        kernel_constraint=constraint
    )

    return tensor_in + projectionlayer






