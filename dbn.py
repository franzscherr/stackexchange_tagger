#!/usr/bin/python
# __________________________________________________________________________________________________
# Implementation of a Deep Belief Network
#

import numpy as np
import tensorflow as tf
import logging

from rbm import RBM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DBN(object):
    def __init__(self, layer_sizes, input_tensor, learning_rate=.001, hidden_tensor=None, n_fantasy_states=100, persistent_contrastive_divergence=True, constrastive_divergence_level=1):
        self.layer_sizes = layer_sizes
        self.machines = []

        current_input = input_tensor
        current_input_size = int(input_tensor.get_shape()[-1])
        for i, size in enumerate(layer_sizes):
            rbm = RBM(current_input_size, size, current_input, learning_rate=learning_rate,
                      n_fantasy_states=n_fantasy_states, name='dbn_{}'.format(i), persistent_contrastive_divergence=persistent_contrastive_divergence, contrastive_divergence_level=constrastive_divergence_level)
            current_input_size = size
            current_input = rbm.hidden_state_plus
            self.machines.append(rbm)
            # def __init__(self, n_features, n_hidden, visible_placeholder=None, contrastive_divergence_level=1, learning_rate=1e-3,
            #              persistent_contrastive_divergence=True, n_fantasy_states=100, name='rbm'):

        if hidden_tensor is None:
            self.hidden_tensor = tf.placeholder(tf.float32, shape=[None, layer_sizes[-1]])
        else:
            self.hidden_tensor = hidden_tensor
        current_hidden = self.hidden_tensor
        self.hidden_tensor = self.hidden_tensor
        for i in range(len(layer_sizes)):
            # dirty hack to connect back to front for sampling
            # TODO: Make this better
            index = len(layer_sizes) - 1 - i
            rbm = self.machines[index]
            with tf.name_scope(rbm.name):
                visible_prop = tf.nn.sigmoid(tf.matmul(current_hidden, rbm.w, transpose_b=True) + rbm.b_v)
                rbm.visible_from_hidden = RBM._sample_from(visible_prop, 'visible_from_hidden')
                current_hidden = rbm.visible_from_hidden

    def train(self, session, training_iterations, training_samples, batch_size=100, saver=None, save_path=None,
              testing_samples=None, testing_batch_size=None):
        reconstruction_errors = []
        if testing_samples is None:
            testing_samples = training_samples
        if testing_batch_size is None:
            testing_batch_size = batch_size
        logger.info('start training with {} samples'.format(training_samples.shape[0]))
        for i, n_iterations in enumerate(training_iterations):
            if i >= len(self.machines):
                break
            rbm = self.machines[i]
            reconstruction_errors_layer = []
            for j in range(n_iterations):
                batch_indices = np.random.randint(0, training_samples.shape[0], size=batch_size)
                batch = training_samples[batch_indices]
                if hasattr(batch, 'todense'):
                    batch = batch.todense()
                session.run([rbm.update_b_v, rbm.update_w, rbm.update_b_h],
                            feed_dict={self.machines[0].visible_placeholder: batch})
                if j % 20 == 0:
                    batch_indices = np.random.randint(0, training_samples.shape[0], size=testing_batch_size)
                    testing_batch = testing_samples[batch_indices]
                    if hasattr(testing_batch, 'todense'):
                        testing_batch = testing_batch.todense()
                    reconstruction_error = session.run(rbm.reconstruction_error,
                                                       feed_dict={self.machines[0].visible_placeholder: testing_batch})
                    reconstruction_errors_layer.append(reconstruction_error)
                    logger.info('reconstruction error in layer {} in iteration {} of {} is {:.4f}'.
                                format(i, j, n_iterations, reconstruction_error))
                if j % 500 == 0 and saver is not None:
                    saver.save(session, save_path=save_path)
                    logger.info('saved model')
            reconstruction_errors.append(reconstruction_errors_layer)
        logger.info('training done')
        return reconstruction_errors

    def sample(self, session, n_samples=1, hidden_seed_binary=True):
        n_last_hidden = self.machines[-1].n_hidden
        if hidden_seed_binary:
            hidden = np.random.randint(0, 2, size=(n_samples, n_last_hidden))
        else:
            hidden = np.random.rand(n_samples, n_last_hidden)
        return session.run(self.machines[0].visible_from_hidden, feed_dict={self.hidden_tensor: hidden})

    def sample_hidden(self, session, visible_states):
        return session.run(self.machines[-1].hidden_state_plus,
                           feed_dict={self.machines[0].visible_placeholder: visible_states})


def main():
    import sklearn.datasets as datasets

    digits = datasets.load_digits()

    features = np.array(digits.data >= 8, dtype=np.float)
    target = digits.target
    n_features = features.shape[1]
    print(n_features)
    layer_sizes = [200, 100, 10]
    inp = tf.placeholder(dtype=tf.float32, shape=[None, n_features])
    dbn = DBN(layer_sizes, inp, learning_rate=1e-3, n_fantasy_states=500)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dbn.train(sess, [20000, 10000, 10000], features, batch_size=500)

        import matplotlib.pyplot as plt
        for i in range(20):
            sample = dbn.sample(sess)
            plt.imshow(sample.reshape([8, 8]), cmap='gray')
            plt.show()


# def main():
#     inp = tf.placeholder(dtype=tf.float32, shape=[None, 10])
#     hid = tf.placeholder(dtype=tf.float32, shape=[None, 5])
#     dbn = DBN([5, 5], inp, hidden_tensor=hid, learning_rate=.002, n_fantasy_states=500)
#     train_data = np.random.randint(0, 2, size=(1000, 5))
#     train_data = np.hstack((train_data, np.zeros((1000, 5))))
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         dbn.train(sess, [5000, 5000], train_data, batch_size=500)
#         # print(sess.run(dbn.machines[-1].hidden_state_plus, feed_dict={inp: np.random.randn(10, 10)}))
#         print(dbn.sample(sess, n_samples=10))


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger.info('START')
    main()
