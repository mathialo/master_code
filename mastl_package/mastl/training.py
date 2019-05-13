import mastl._iohelp as io

import tensorflow as tf
import numpy as np

import os
import time
import datetime
import shutil


def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
    """
    Adapted version of an answer by StackOverflow user Lior at
    https://stackoverflow.com/questions/47997203/tensorflow-restore-if-present


    Args:
        session:        session object
        save_file:      save file to load
        graph:          graph to act upon
    """
    session.run(tf.global_variables_initializer())

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


def train_denoiser(config):
    try:
        try:
            dataset = config['dataset']
            assert dataset is not None

            io.log("Data loaded: {} training samples, {} testing samples".format(
                len(dataset.data_files),
                len(dataset.test_files)
            ))
        except (AssertionError, KeyError):
            io.log("Could not load dataset from configure.py", "error")
            return

        if 'output' in config:
            dirname = config['output']
        else:
            dirname = str(int(time.time()))

        io.log("Creating save dir ./{}".format(dirname))
        os.mkdir(dirname)

        if 'restore' in config:
            restore = config['restore']
            noticefile = open("{}/IMPORTANT_NOTICE".format(dirname), "a")
            noticefile.write("This model is a continuation of {}. ".format(restore))
            noticefile.write("You should disregard the model specifications in "
                             "configure.py, and look there instead. The training "
                             "specifications however, are correct.\n\n")
            noticefile.close()

        try:
            model_datatype = config['model_datatype']
            assert model_datatype is not None

        except:
            io.log("Could not read mode_datatype from configure.py", "warn")
            io.log("Defaulting to tf.float32", "detail")

            model_datatype = tf.float32

        try:
            x, y, channels_num = config['x_size'], config['y_size'], config['channels_num']
        except:
            io.log("Could not read x, y or c size from configure.py", "warn")
            io.log("Defaulting to None", "detail")

            x, y, channels_num = None, None, None

        if 'restore' in config:
            io.log("Loading previous sampling pattern")
            sampling_mask = np.load("{}/pattern.npy".format(restore))

        else:
            io.log("Loading sampling pattern from config")
            try:
                sampling_mask = config['sampling_mask']
                assert sampling_mask is not None
            except (AssertionError, KeyError):
                io.log("Could not read sampling pattern from configure.py", "warn")
                io.log("Defaulting to gaussian sampling at 20 % sampling rate", "detail")

                if x is None or y is None:
                    io.log("Can't create sampling pattern, x or y is None", "error")
                    return

                from tools.patterns import gaussian_sampling

                sampling_mask = gaussian_sampling(x, y, int(x * y * 0.2))

            np.save("{}/pattern.npy".format(dirname), sampling_mask)

        io.log("Sampling rate: {}".format(np.mean(sampling_mask)), "detail")
        sampling_mask = np.fft.fftshift(sampling_mask)


        def sample_op(input):
            fft_input = np.fft.fft2(input)
            fft_input[np.invert(sampling_mask)] = 0
            return fft_input


        io.log("Creating TF Graph")

        try:
            batch_size = config['batch_size']
            assert batch_size is not None

        except AssertionError:
            io.log("Could not read batch size from configure.py", "warn")
            io.log("Defaulting to 10", "detail")

            batch_size = 10

        input = tf.placeholder(
            model_datatype,
            shape=[batch_size, y, x, channels_num]
        )

        is_training = tf.placeholder(tf.bool)

        try:
            output = config['model'](input, is_training)
        except Exception as e:
            io.log("Could not create model from configure.py", "error")
            io.log("Error: {}".format(str(e)), "detail")

            return

        true_output = tf.placeholder(model_datatype, shape=[batch_size, y, x, channels_num])

        try:
            loss = config['loss'](output, true_output)
        except:
            io.log("Could not get loss function from configure.py", "error")
            return

        if 'restore' in config:
            io.log("Loading train_losses and updating training status")
            losses = np.load("{}/train_losses.npy".format(restore)).tolist()
            start_train = len(losses)

            # Update dataset status
            dataset.epoch = start_train * batch_size // len(
                dataset)
            dataset.index = start_train * batch_size % len(
                dataset)

        else:
            losses = []
            start_train = 0

        try:
            train_steps = config['train_steps']
            assert train_steps is not None

        except AssertionError:
            io.log("Could not read train_steps from configure.py", "warn")
            io.log("Defaulting to 100", "detail")

            train_steps = 100

        try:
            optimizer = config['optimizer']
            assert optimizer is not None
        except:
            io.log("Could not read optimizer from configure.py", "warn")
            io.log("Defaulting to tf.train.GradientDecentOptimizer", "detail")

            optimizer = tf.train.GradientDescentOptimizer(0.001)

        train_step = optimizer.minimize(loss, var_list=tf.global_variables(), colocate_gradients_with_ops=True)

        # Initialize saver
        saver = tf.train.Saver(max_to_keep=None)

        io.log("Planning to do {} training steps in total, ie {} epochs".format(train_steps,
                                                                                train_steps * batch_size // (
                                                                                    len(
                                                                                        dataset))))

        # Start TF session
        io.log("Initializing TF Session")
        sessconfig = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=sessconfig) as sess:
            if 'restore' in config:
                saver.restore(sess, "{}/last_run.ckpt".format(config['restore']))
            else:
                sess.run(tf.global_variables_initializer())

            io.log("Beginning training")
            time_start = time.time()

            try:
                for i in range(start_train, start_train + train_steps):
                    if i == start_train:
                        print(
                            "\r%.1f %%    Loss: %-10g    Time elapsed: %-10s    ETA: %-10s       " % (
                                0,
                                np.nan,
                                "-",
                                "-"
                            ),
                            end=""
                        )

                    else:
                        if i % 20 == 0:
                            print(
                                "\rSTATUS:  %.1f %% done  Avg loss: %-10g  Training progress: %s %s" % (
                                    (i - start_train) * 100 / train_steps,
                                    np.mean(np.array(losses[i - 20:i - 1])),
                                    "{} / {} in epoch {}".format(dataset.index,
                                                                 len(dataset),
                                                                 dataset.epoch),
                                    " "*40
                                )
                            )

                        elapsed = int(time_now - time_start)
                        eta = int(
                            elapsed / (i - start_train + 1) * (train_steps - i + start_train))
                        print(
                            "\r%.1f %%    Loss: %-10g    Time elapsed: %-10s    ETA: %-10s     Training progress: %s  " % (
                                (i - start_train) * 100 / train_steps,
                                losses[i - 1],
                                str(datetime.timedelta(seconds=elapsed)),
                                str(datetime.timedelta(seconds=eta)),
                                "{} / {} in epoch {}".format(dataset.index,
                                                             len(dataset),
                                                             dataset.epoch)
                            ),
                            end=""
                        )
                        if i % 500 == 0:
                            print("\nSaving at {} steps".format(i))
                            saver.save(sess, "{}/at_{}_steps.ckpt".format(dirname, i))

                    dataset.ready_next_batch(batch_size)
                    dataset.sample_loaded_batch(sample_op)
                    data, sampled = dataset.get_next_batch()

                    sess.run(train_step, feed_dict={input:       sampled,
                                                    true_output: data,
                                                    is_training: True
                                                    })
                    losses.append(sess.run(loss, feed_dict={input:       sampled,
                                                            true_output: data,
                                                            is_training: False
                                                            }))

                    time_now = time.time()

                print("\r100.0 %%  Loss: %g%s" % (
                    losses[start_train + train_steps - 1], " " * 80))
                elapsed = time_now - time_start
                io.log("Total time: {}".format(str(datetime.timedelta(seconds=elapsed))))

            except KeyboardInterrupt:
                print()
                io.log("Stopping training")

                noticefile = open("{}/IMPORTANT_NOTICE".format(dirname), "a")
                noticefile.write("Training was aborted after {} steps.\n\n".format(i))
                noticefile.close()

            io.log("Status: {} / {} in epoch {}".format(dataset.index,
                                                        len(dataset),
                                                        dataset.epoch))

            io.log("Running model on train data")
            os.mkdir("{}/{}".format(dirname, "train_results"))

            dataset.ready_next_batch(batch_size)
            dataset.sample_loaded_batch(sample_op)
            data, sampled = dataset.get_next_batch()

            recovery = sess.run(output, feed_dict={input: sampled, is_training: False})
            for j in range(batch_size):
                num = i * batch_size + j
                np.save(
                    "{}/{}/{:03}_sample.npy".format(dirname, "train_results", (num)),
                    np.squeeze(np.abs(data[j]))
                )
                np.save(
                    "{}/{}/{:03}_adjoint.npy".format(dirname, "train_results", (num)),
                    np.abs(np.fft.ifft2(np.squeeze(sampled[j])))
                )
                np.save(
                    "{}/{}/{:03}_recovery.npy".format(dirname, "train_results", (num)),
                    np.squeeze(np.abs(recovery[j]))
                )



            io.log("Running model on test data")
            os.mkdir("{}/{}".format(dirname, "test_results"))

            test_set, sampled_test_set = dataset.get_test_set(sample_op=sample_op)

            for i in range(len(test_set) // batch_size):
                print(
                    "\rRunning... %.2f %%" % (
                            i / (len(test_set) // batch_size) * 100),
                    end=""
                )

                data = test_set[i * batch_size: (i + 1) * batch_size]
                sampled = sampled_test_set[i * batch_size: (i + 1) * batch_size]

                recovery = sess.run(output, feed_dict={input: sampled, is_training: False})

                for j in range(batch_size):
                    num = i * batch_size + j
                    np.save(
                        "{}/{}/{:03}_sample.npy".format(dirname, "test_results", (num)),
                        np.squeeze(np.abs(data[j]))
                    )
                    np.save(
                        "{}/{}/{:03}_adjoint.npy".format(dirname, "test_results", (num)),
                        np.abs(np.fft.ifft2(np.squeeze(sampled[j])))
                    )
                    np.save(
                        "{}/{}/{:03}_recovery.npy".format(dirname, "test_results", (num)),
                        np.squeeze(np.abs(recovery[j]))
                    )
            print("\rRunning... 100 %     ")

            np.save("{}/train_losses.npy".format(dirname), np.array(losses))
            saver.save(sess, "{}/last_run.ckpt".format(dirname))

    except BaseException as e:
        print()
        io.log("Error occured!", "error")
        # io.log("Cleaning up save dir", "detail")
        # shutil.rmtree("./{}".format(dirname))

        raise e
