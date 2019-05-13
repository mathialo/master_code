import foolbox
import tensorflow as tf
import numpy as np
import sys

labels = [
	"airplane",
	"automobile",
	"bird",
	"cat",
	"deer",
	"dog",
	"frog",
	"horse",
	"ship",
	"truck"
]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

img_size = (32, 32, 3)
batch_size = 1

input_layer = tf.placeholder(tf.float32, shape=(batch_size, *img_size))
onehot_label = tf.placeholder(tf.float32, shape=(batch_size, 10))

last_layer = input_layer

last_layer = tf.layers.conv2d(
    inputs=last_layer,
    kernel_size=(5, 5),
    filters=32,
    strides=(1, 1),
    padding="SAME",
    activation=None,
    kernel_initializer=None,
    kernel_constraint=None,
    
)/5
print("conv", last_layer.shape)
last_layer = tf.layers.conv2d(
    inputs=last_layer,
    kernel_size=(3, 3),
    filters=32,
    strides=(1, 1),
    padding="SAME",
    activation=None,
    kernel_initializer=None,
    kernel_constraint=None,
    
)
print("conv", last_layer.shape)
last_layer = tf.layers.max_pooling2d(
    last_layer,
    pool_size=(2, 2),
    strides=(2, 2)
)/3
print("maxpool", last_layer.shape)
last_layer = tf.nn.relu(last_layer)
last_layer = tf.layers.dropout(last_layer, rate=0.15, training=False)
last_layer = tf.layers.conv2d(
    inputs=last_layer,
    kernel_size=(3, 3),
    filters=64,
    strides=(1, 1),
    padding="SAME",
    activation=tf.nn.relu,
    kernel_initializer=None,
    kernel_constraint=None,
    
)/3
print("conv", last_layer.shape)
last_layer = tf.layers.dropout(last_layer, rate=0.15, training=False)
last_layer = tf.layers.conv2d(
    inputs=last_layer,
    kernel_size=(3, 3),
    filters=64,
    strides=(1, 1),
    padding="SAME",
    activation=tf.nn.relu,
    kernel_initializer=None,
    kernel_constraint=None,
    
)/3
print("conv", last_layer.shape)
last_layer = tf.layers.dropout(last_layer, rate=0.15, training=False)
last_layer = tf.layers.conv2d(
    inputs=last_layer,
    kernel_size=(3, 3),
    filters=64,
    strides=(1, 1),
    padding="SAME",
    activation=None,
    kernel_initializer=None,
    kernel_constraint=None,
    
)/3
print("conv", last_layer.shape)
last_layer = tf.layers.dropout(last_layer, rate=0.15, training=False)
last_layer = tf.layers.max_pooling2d(last_layer, pool_size=(2, 2), strides=(2, 2))
print("maxpool", last_layer.shape)
last_layer = tf.nn.relu(last_layer)
last_layer = tf.layers.conv2d(
    inputs=last_layer,
    kernel_size=(3, 3),
    filters=128,
    strides=(1, 1),
    padding="SAME",
    activation=None,
    kernel_initializer=None,
    kernel_constraint=None,
    
)/3
print("conv", last_layer.shape)
last_layer = tf.layers.dropout(last_layer, rate=0.15, training=False)
last_layer = tf.layers.conv2d(
    inputs=last_layer,
    kernel_size=(3, 3),
    filters=128,
    strides=(1, 1),
    padding="SAME",
    activation=None,
    kernel_initializer=None,
    kernel_constraint=None,
    
)/3
print("conv", last_layer.shape)
last_layer = tf.layers.dropout(last_layer, rate=0.15, training=False)


last_layer = tf.layers.dropout(last_layer, rate=0.15, training=False)
last_layer = tf.layers.flatten(last_layer)
last_layer = tf.layers.dense(
    last_layer,
    1024,
    activation=tf.nn.relu,
    kernel_initializer=None,
    kernel_constraint=None,
)
print("dense", last_layer.shape)
last_layer = tf.layers.dropout(last_layer, rate=0.15, training=False)
last_layer = tf.layers.dense(
    last_layer,
    1024,
    activation=tf.nn.tanh,
    kernel_initializer=None,
    kernel_constraint=None,
)
print("dense", last_layer.shape)
last_layer = tf.layers.dropout(last_layer, rate=0.15, training=False)
last_layer = tf.layers.dense(
    tf.layers.flatten(last_layer),
    10,
    kernel_initializer=None,
    kernel_constraint=None,
)
print("dense", last_layer.shape)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=last_layer, labels=onehot_label))
predicted = tf.argmax(last_layer, axis=1)

sess = tf.Session()

num_tests = int(sys.argv[2])
test = 0
image_num = 0

mean_snr = 0

with sess.as_default():
	fool_model = foolbox.models.TensorFlowModel(input_layer, last_layer, bounds=(0, 255))
	tf.train.Saver().restore(sess, "{}/last_run.ckpt".format(sys.argv[1]))
	attack = foolbox.attacks.DeepFoolAttack(fool_model, criterion=foolbox.criteria.Misclassification())


	while test < num_tests:
		image, label = x_test[image_num,:,:,:], y_test[image_num][0]
		
		# Skip where label is predicted wrong
		if sess.run(predicted, feed_dict={input_layer: np.expand_dims(image, 0)})[0] != label:
			image_num += 1
			continue

		adv = attack(image, label)
		if adv is None:
			print("Warn: None returned")
			image_num += 1
			continue

		np.save("{:03d}_original.npy".format(test), np.squeeze(image))
		np.save("{:03d}_perturbed.npy".format(test), np.squeeze(adv))

		label_orig = sess.run(predicted, feed_dict={input_layer: np.expand_dims(image, 0)})[0]
		label_adv = sess.run(predicted, feed_dict={input_layer: np.expand_dims(adv, 0)})[0]

		print("{}: Original label:   {}".format(test, labels[label_orig]))
		print("{}: Perturbed label:  {}".format(test, labels[label_adv]))

		snr = 10*np.log10(np.linalg.norm(image)/np.linalg.norm(adv-image))
		mean_snr += snr/num_tests
		print("{}: SNR:              {:.3f}".format(test, snr))

		image_num += 1
		test += 1

print("Mean SNR: {}".format(mean_snr))

sess.close()