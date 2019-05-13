import tensorflow as tf
import numpy as np
import sigpy.imaging as img

import datetime
import time

import os

import parsnet.constraints

# from tools.dataset import DataSet


def augment(data):
    flip_these = np.random.permutation(len(data))[0:np.random.randint(0, len(data))]
    for ind in flip_these:
        data[ind, :, :, :] = np.flip(data[ind,:,:,:], axis=np.random.randint(2))

    add_noise = np.random.permutation(len(data))[0:np.random.randint(0, len(data)//2)]
    for ind in add_noise:
        data[ind, :, :, :] += np.random.normal(0, 0.02, data[ind].shape)

def data_loader(img_name):
    lab_name = img_name.replace("img", "onehot")

    return {'data': np.load(img_name), 'label': np.load(lab_name)}


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Convert to one-hot
onehot_train = np.zeros([len(y_train), 10])
for i, lab in enumerate(y_train):
    onehot_train[i, lab] = 1

onehot_test = np.zeros([len(y_test), 10])
for i, lab in enumerate(y_test):
    onehot_test[i, lab] = 1


img_size = (32, 32, 3)
batch_size = 512
train_epochs = 50
beta = 0.001


input_layer = tf.placeholder(tf.float32, shape=(batch_size, *img_size))
onehot_label = tf.placeholder(tf.float32, shape=(batch_size, 10))

is_training = tf.placeholder(tf.bool)

print("Architecture:")
last_layer = input_layer
print("model_input", last_layer.shape)


last_layer = tf.layers.conv2d(
    inputs=last_layer,
    kernel_size=(5, 5),
    filters=64,
    strides=(1, 1),
    padding="SAME",
    activation=None,
    kernel_initializer=tf.initializers.orthogonal(),
    kernel_constraint=parsnet.constraints.tight_frame(beta, 3),
    name="convlayer{}".format(1)
)
print("conv", last_layer.shape)
last_layer = tf.layers.max_pooling2d(
    last_layer,
    pool_size=(2, 2),
    strides=(2, 2)
)
print("maxpool", last_layer.shape)
last_layer = tf.nn.relu(last_layer)
last_layer = tf.layers.dropout(last_layer, rate=0.25, training=is_training)
last_layer = tf.layers.conv2d(
    inputs=last_layer,
    kernel_size=(3, 3),
    filters=64,
    strides=(1, 1),
    padding="SAME",
    activation=tf.nn.relu,
    kernel_initializer=tf.initializers.orthogonal(),
    kernel_constraint=parsnet.constraints.tight_frame(beta, 3),
    name="convlayer{}".format(2)
)
print("conv", last_layer.shape)
last_layer = tf.layers.dropout(last_layer, rate=0.25, training=is_training)
last_layer = tf.layers.conv2d(
    inputs=last_layer,
    kernel_size=(3, 3),
    filters=32,
    strides=(1, 1),
    padding="SAME",
    activation=tf.nn.relu,
    kernel_initializer=tf.initializers.orthogonal(),
    kernel_constraint=parsnet.constraints.tight_frame(beta, 3),
    name="convlayer{}".format(3)
)
print("conv", last_layer.shape)
last_layer = tf.layers.dropout(last_layer, rate=0.25, training=is_training)
last_layer = tf.layers.conv2d(
    inputs=last_layer,
    kernel_size=(3, 3),
    filters=32,
    strides=(1, 1),
    padding="SAME",
    activation=tf.nn.relu,
    kernel_initializer=tf.initializers.orthogonal(),
    kernel_constraint=parsnet.constraints.tight_frame(beta, 3),
    name="convlayer{}".format(4)
)
print("conv", last_layer.shape)
last_layer = tf.layers.dropout(last_layer, rate=0.25, training=is_training)
last_layer = tf.layers.max_pooling2d(last_layer, pool_size=(2, 2), strides=(2, 2))
print("maxpool", last_layer.shape)
last_layer = tf.layers.dropout(last_layer, rate=0.15, training=is_training)
last_layer = tf.layers.flatten(last_layer)
last_layer = tf.layers.dense(
    last_layer,
    64,
    activation=tf.nn.relu,
    kernel_initializer=tf.initializers.orthogonal(),
    kernel_constraint=parsnet.constraints.tight_frame(beta, 3)
)
print("dense", last_layer.shape)
last_layer = tf.layers.dropout(last_layer, rate=0.25, training=is_training)
last_layer = tf.layers.dense(
    last_layer,
    64,
    activation=tf.nn.tanh,
    kernel_initializer=tf.initializers.orthogonal(),
    kernel_constraint=parsnet.constraints.tight_frame(beta, 3)
)
print("dense", last_layer.shape)
last_layer = tf.layers.dropout(last_layer, rate=0.25, training=is_training)
last_layer = tf.layers.dense(
    tf.layers.flatten(last_layer),
    10,
    kernel_initializer=tf.initializers.orthogonal(),
    kernel_constraint=parsnet.constraints.tight_frame(beta, 3),
)
print("dense", last_layer.shape)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=last_layer, labels=onehot_label))

trainer = tf.train.AdamOptimizer()
train_op = trainer.minimize(loss, var_list=tf.global_variables())

predicted = tf.argmax(last_layer, axis=1)

total_num_steps = len(y_train)//batch_size*train_epochs
print("Will do {} epochs of size {}, totaling in {} variable updates.".format(train_epochs, len(y_train), total_num_steps))

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
steps = []

saver = tf.train.Saver(max_to_keep=None)

start_time = time.time()
with tf.Session() as sess:
    if os.path.exists("./initialized.ckpt.meta") and False:
        print("Restoring previous initialization")
        saver.restore(sess, "./initialized.ckpt")
    else:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, "./initialized.ckpt")


    for epoch in range(train_epochs):
        train_indexes = np.random.permutation(len(y_train))

        for step in range(len(y_train)//batch_size):

            # dataset.ready_next_batch(batch_size)
            # data, labels = dataset.get_next_batch()
            start, end = step*batch_size, (step+1)*batch_size
            data = x_train[start:end]
            labels = onehot_train[start:end]
            data = img.map_to_01(data)
            augment(data)

            sess.run(train_op, feed_dict={input_layer: data, onehot_label: labels, is_training: True})

            total_step = len(y_train)//batch_size*epoch + step

            if total_step % 10 == 0:
                loss_value, logits, predicted_value = sess.run([loss, last_layer, predicted], feed_dict={input_layer: data, onehot_label: labels, is_training: False})
                actual = np.argmax(labels, axis=1)

                print("Status: {:.1f} %. Train loss: {:.2f}. Train error: {:.1f} % ".format(total_step / total_num_steps*100, loss_value, np.mean(predicted_value != actual)*100), end="")
                train_losses.append(loss_value)
                train_accuracies.append(np.mean(predicted_value == actual)*100)
                steps.append(total_step)

                val_indexes = np.random.permutation(len(y_test))[0:512]
                data = x_test[val_indexes]
                labels = onehot_test[val_indexes]
                data = img.map_to_01(data)

                loss_value, logits, predicted_value = sess.run([loss, last_layer, predicted], feed_dict={input_layer: data, onehot_label: labels, is_training: False})
                actual = np.argmax(labels, axis=1)
                print("Validation loss: {:.2f}. Validation error: {:.1f} % ".format(loss_value, np.mean(predicted_value != actual) * 100))
                val_losses.append(loss_value)
                val_accuracies.append(np.mean(predicted_value == actual) * 100)

            if total_step % 500 == 0 and total_step != 0:
                print("Saving at {} global steps".format(total_step))
                saver.save(sess, "./at_{}_gobal_steps.ckpt".format(total_step))


    saver.save(sess, "./last_run.ckpt")
    np.save("train_losses.npy", np.array(train_losses))
    np.save("train_accuracies.npy", np.array(train_accuracies))
    np.save("val_losses.npy", np.array(val_losses))
    np.save("val_accuracies.npy", np.array(val_accuracies))
    np.save("steps.npy", np.array(steps))

print("Total time taken: {}".format(str(datetime.timedelta(seconds=(time.time() - start_time)))))