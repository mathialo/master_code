import tensorflow as tf


def l1(point1, point2):
    return tf.reduce_sum(tf.abs(point1 - point2)) / int(point1.shape[0])


def l2(point1, point2):
    return tf.abs(tf.reduce_sum(tf.square(point1 - point2))) / int(point1.shape[0])


def linf(point1, point2):
    return tf.reduce_max(tf.abs(point1 - point2))

