import tensorflow as tf
import numpy as np
import sys


def create_model(input_node):
    layer1 = tf.layers.conv2d(
        input_node,
        16,
        (3, 3),
        padding="SAME",
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.he_normal()
    )
    layer2 = tf.layers.conv2d(
        layer1,
        32,
        (3, 3),
        padding="SAME",
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.he_normal()
    )
    layer3 = tf.layers.conv2d(
        layer2,
        32,
        (3, 3),
        padding="SAME",
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.he_normal()
    )
    layer4 = tf.layers.conv2d(
        layer3,
        16,
        (3, 3),
        padding="SAME",
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.he_normal()
    )
    layer5 = tf.layers.conv2d(
        layer4,
        1,
        (3, 3),
        padding="SAME",
        activation=tf.nn.tanh,
        kernel_initializer=tf.keras.initializers.he_normal()
    )

    return input_node + layer5


def add_noise(data, amount=0.1):
    return data+np.random.normal(0, amount, data.shape)


def get_data():
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    return x_train/255, x_test/255



def train():
    x_train, _ = get_data()
    x_train = np.expand_dims(x_train, -1)

    input_node = tf.placeholder(tf.float32, shape=(None, x_train.shape[1], x_train.shape[2], 1))
    expected_node = tf.placeholder(tf.float32, shape=(None, x_train.shape[1], x_train.shape[2], 1))

    output_node = create_model(input_node)

    loss_node = tf.reduce_mean(tf.squared_difference(output_node, expected_node))
    optimizer_step = tf.train.AdamOptimizer().minimize(loss_node)

    saver = tf.train.Saver()

    batch_size = 100
    epochs = 200

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            x_train = np.random.permutation(x_train)

            print("Epoch {} / {}".format(epoch+1, epochs))

            for step in range(x_train.shape[0] // batch_size):
                start, end = step * batch_size, (step + 1) * batch_size
                data = x_train[start:end]

                sess.run(optimizer_step, feed_dict={input_node: add_noise(data), expected_node: data})

                if step % 100 == 0:
                    loss = sess.run(loss_node, feed_dict={input_node: add_noise(data), expected_node: data})
                    print("Loss: %g" % loss)

        saver.save(sess, "./trained_denoiser.ckpt")


def inference():
    _, x_test = get_data()
    x_test = np.expand_dims(x_test, -1)

    input_node = tf.placeholder(tf.float32, shape=(None, x_test.shape[1], x_test.shape[2], 1))
    expected_node = tf.placeholder(tf.float32, shape=(None, x_test.shape[1], x_test.shape[2], 1))

    output_node = create_model(input_node)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "./trained_denoiser.ckpt")

        data = x_test[0:8]
        noisy = add_noise(data)
        denoised = sess.run(output_node, feed_dict={input_node: noisy})

        np.save("original.npy", data)
        np.save("noisy.npy", noisy)
        np.save("denoised.npy", denoised)


def figure():
    original = np.squeeze(np.load("original.npy"))
    noisy = np.squeeze(np.load("noisy.npy"))
    denoised = np.squeeze(np.load("denoised.npy"))

    image = np.vstack([
        np.hstack(original),
        np.hstack(noisy),
        np.hstack(denoised),
    ])

    from sigpy.imaging import display
    display(image, filename="mnist_denoise.png", show=False)




if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("No command given")
        sys.exit()


    for arg in sys.argv[1:]:
        if arg == "train":
            train()

        elif arg == "inference":
            inference()

        elif arg == "figure":
            figure()

