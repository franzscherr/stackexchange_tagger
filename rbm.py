#!/usr/bin/python
# __________________________________________________________________________________________________
# Implementation of a restricted boltzmann machine in tensorflow
#

import numpy as np
import tensorflow as tf
import logging

from adam import ADAMOptimizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RBM(object):
    def __init__(self, n_features, n_hidden, visible_placeholder=None, contrastive_divergence_level=1, learning_rate=1e-3,
                 persistent_contrastive_divergence=True, n_fantasy_states=100, name='rbm'):
        """
        A restricted boltzmann machine implemented in tensorflow
        
        :param n_features: The input dimensionality
        :param n_hidden: The hidden layer size
        :param visible_placeholder: Optional. Is used for chaining. 
        A tensorflow tensor that represents the input [batch x n_features].
        :param contrastive_divergence_level: How often to sample to get the model distribution
        :param learning_rate: The learning rate for gradient ascent on the likelihood
        :param persistent_contrastive_divergence: Use a persistent model state for model distribution. Replaces
        contrastive divergence
        :param fantasy_states: Number of fantasy samples to keep in the persistent mode.
        :param name: tensorflow name of this model
        """
        if visible_placeholder is None:
            self.visible_placeholder = tf.placeholder(tf.float32, shape=(None, n_features))
        else:
            self.visible_placeholder = visible_placeholder
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.contrastive_divergence_level = contrastive_divergence_level
        self.persistent_contrastive_divergence = persistent_contrastive_divergence
        self.name = name

        with tf.name_scope(name):
            # variables
            self.w = tf.Variable(tf.random_normal((self.n_features, self.n_hidden), mean=0.0, stddev=np.sqrt(6./(self.n_features + self.n_hidden))), name='weights')
            self.b_h = tf.Variable(tf.random_normal((self.n_hidden,), mean=0.0, stddev=0.1), name='hidden_bias')
            self.b_v = tf.Variable(tf.random_normal((self.n_features,), mean=0.0, stddev=0.1), name='visible_bias')

            # layers
            self.hidden_data_probability = tf.nn.sigmoid(tf.matmul(self.visible_placeholder, self.w) + self.b_h)
            self.hidden_state_plus = RBM._sample_from(self.hidden_data_probability, 'hidden_state_plus')

            # sampling
            self.hidden_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.n_hidden))
            visible_prop = tf.nn.sigmoid(tf.matmul(self.hidden_placeholder, self.w, transpose_b=True) + self.b_v)
            self.visible_from_hidden = RBM._sample_from(visible_prop, 'visible_from_hidden')

            # persistent CD
            if self.persistent_contrastive_divergence:
                # keep fantasy states that sample from the model distribution
                self.visible_state_persistent = tf.Variable(
                    initial_value=tf.zeros((n_fantasy_states, self.n_features), dtype=tf.float32),
                    trainable=False,
                    name='visible_model_sampling_persistent_state')
                self.hidden_state_persistent = tf.Variable(
                    initial_value=tf.zeros((n_fantasy_states, self.n_hidden), dtype=tf.float32),
                    trainable=False,
                    name='hidden_model_sampling_persistent_state')

                # reach equilibrium after parameter update by sampling v -> h, h -> v
                current_hidden_state = self.hidden_state_persistent
                current_visible_state = self.visible_state_persistent
                for i in range(self.contrastive_divergence_level):
                    hidden_probability = tf.nn.sigmoid(tf.matmul(current_visible_state, self.w) + self.b_h)
                    current_hidden_state = RBM._sample_from(hidden_probability,
                                                            'hidden_state_minus_{}'.format(i))
                    visible_probability = tf.nn.sigmoid(tf.matmul(current_hidden_state, self.w, transpose_b=True)
                                                        + self.b_v)
                    current_visible_state = RBM._sample_from(visible_probability,
                                                             'visible_state_minus_{}'.format(i))
                self.visible_state_minus = tf.assign(self.visible_state_persistent, current_visible_state,
                                                     name='visible_state_minus')
                self.hidden_state_minus = tf.assign(self.hidden_state_persistent, current_hidden_state,
                                                    name='hidden_state_minus')
            else:
                current_hidden = self.hidden_data_probability

                for i in range(self.contrastive_divergence_level):
                    visible_prop = tf.nn.sigmoid(tf.matmul(current_hidden, self.w, transpose_b=True) + self.b_v)
                    self.visible_state_minus = RBM._sample_from(visible_prop, 'visible_state_minus_{}'.format(i))

                    hidden_prop = tf.nn.sigmoid(tf.matmul(self.visible_state_minus, self.w) + self.b_h)
                    self.hidden_state_minus = RBM._sample_from(hidden_prop, 'hidden_state_minus_{}'.format(i))

            # train operations: contrastive divergence
            gradient_w = (tf.matmul(self.visible_placeholder, self.hidden_state_plus, transpose_a=True) - tf.matmul(
                self.visible_state_minus, self.hidden_state_minus,
                transpose_a=True)) / tf.cast(tf.shape(self.visible_placeholder)[0], dtype=tf.float32)
            gradient_b_h = tf.reduce_mean(self.hidden_state_plus, reduction_indices=0) - tf.reduce_mean(
                self.hidden_state_minus, reduction_indices=0)
            gradient_b_v = tf.reduce_mean(self.visible_placeholder, reduction_indices=0) - tf.reduce_mean(
                self.visible_state_minus, reduction_indices=0)

            self.reconstruction_error = tf.reduce_mean(tf.reduce_sum(
                (self.visible_placeholder - self.visible_state_minus)**2, axis=-1))
            #
            # # update
            # self.update_w = tf.assign_add(self.w, self.learning_rate * gradient_w)
            # self.update_b_h = tf.assign_add(self.b_h, self.learning_rate * gradient_b_h)
            # self.update_b_v = tf.assign_add(self.b_v, self.learning_rate * gradient_b_v)
            beta = .2
            self.update_w = ADAMOptimizer(self.w, minimize=False, beta1=beta, beta2=beta,
                                          learning_rate=self.learning_rate).train_step(gradient_w)
            self.update_b_h = ADAMOptimizer(self.b_h, minimize=False, beta1=beta, beta2=beta,
                                            learning_rate=learning_rate).train_step(gradient_b_h)
            self.update_b_v = ADAMOptimizer(self.b_v, minimize=False, beta1=beta, beta2=beta,
                                            learning_rate=learning_rate).train_step(gradient_b_v)
            self.train_step = [self.update_w, self.update_b_h, self.update_b_v]
            self.logarithmic_energy = tf.reduce_sum(tf.matmul(self.visible_placeholder, self.w) *
                                                    self.hidden_placeholder, axis=-1) + \
                tf.reduce_sum(self.visible_placeholder * self.b_v, axis=-1) + \
                tf.reduce_sum(self.hidden_placeholder * self.b_h, axis=-1)
            self.state_probabilities = tf.nn.softmax(self.logarithmic_energy)

    def _sample_from(vector, name):
        return tf.cast(tf.less(tf.random_uniform(tf.shape(vector)), vector), tf.float32, name=name)

    def train(self, features, n_iterations=1000, batch_size=100, session=None):
        """
        Perform training of the boltzmann machine given an input distribution
        
        :param features: numpy array representing the input distribution [n_samples x n_features]
        :param n_iterations: Number of iterations to train
        :param batch_size: Number of samples to use in one training iteration
        :param session: tensorflow session to use for the training. If None, default will be used.
        """
        if session is None:
            session = tf.get_default_session()
        train_range = range(n_iterations)
        n_samples = features.shape[0]
        for i in train_range:
            random_indices = np.random.randint(0, n_samples, size=batch_size)
            feed_dict = {self.visible_placeholder: features[random_indices]}
            reconstruction_error, _, _, _ = session.run([self.reconstruction_error, self.update_b_h, self.update_w,
                                                         self.update_b_v], feed_dict=feed_dict)
            if i % 20 == 0:
                logger.info('reconstruction error in iteration {} of {} is {}'.format(i, n_iterations,
                                                                                      reconstruction_error))

    def sample_visible(self, n_samples=None, hidden_states=None, session=None):
        """
        Sample visible layer states from the boltzmann machine. Either use hidden_states for initializing the hidden
        layer if given or use a binary random hidden_state.
        
        :param n_samples: Optional. Number of states to sample.
        :param hidden_states: Optional. If given, sample the visible states from these hidden states.
        :param session: tensorflow session to use. If None, use default session.
        :return: An array with visible states. The number of states returned is n_samples if given or the row number of
        hidden_states. If nothing was given, a single sample is returned.
        """
        if session is None:
            session = tf.get_default_session()
        if hidden_states is None:
            hidden_states = np.random.randint(0, 2, size=(n_samples if n_samples is not None else 1, self.n_hidden))
        return session.run(self.visible_from_hidden, feed_dict={self.hidden_placeholder: hidden_states})

    def sample_hidden(self, n_samples=None, visible_states=None, session=None):
        """
        Sample hidden layer states from the boltzmann machine. Either use visible_states for initializing the visible
        layer if given or use a binary random hidden_state.
        
        :param n_samples: Optional. Number of states to sample.
        :param visible_states: Optional. If given, sample the hidden states from these visible states.
        :param session: tensorflow session to use. If None, use default session.
        :return: An array with hidden states. The number of states returned is n_samples if given or the row number of
        visible_states. If nothing was given, a single sample is returned.
        """
        if session is None:
            session = tf.get_default_session()
        if visible_states is None:
            visible_states = np.random.randint(0, 2, size=(n_samples if n_samples is not None else 1, self.n_features))
        return session.run(self.hidden_state_plus, feed_dict={self.visible_placeholder: visible_states})


if __name__ == '__main__':
    import sklearn.datasets as datasets
    import sklearn.svm as svm

    digits = datasets.load_digits()

    features = np.array(digits.data >= 8, dtype=np.float)
    target = digits.target
    n_features = features.shape[1]
    n_hidden = 300
    n_hidden2 = 300
    n_hidden3 = 300
    n_batch = 2000
    n_iterations = 1000

    input_placeholder = tf.placeholder(tf.float32, shape=(None, n_features))

    rbm = RBM(n_features, n_hidden, input_placeholder, name='rbm_1', n_fantasy_states=n_batch,
              persistent_contrastive_divergence=False, contrastive_divergence_level=10, learning_rate=1e-3)
    rbm2 = RBM(n_hidden, n_hidden2, rbm.hidden_state_plus, learning_rate=1e-4, name='rbm_2', n_fantasy_states=n_batch)
    rbm3 = RBM(n_hidden2, n_hidden3, rbm2.hidden_state_plus, learning_rate=1e-4, name='rbm_3', n_fantasy_states=n_batch)

    svc = svm.SVC(kernel='linear')
    svc2 = svm.SVC(kernel='linear')
    svc2.fit(features[:-100], target[:-100])
    direct_svc_score = svc2.score(features[-100:], target[-100:])

    session = tf.Session()
    session.run(tf.initialize_all_variables())
    for i in range(n_iterations):
        random_indices = np.random.randint(0, features.shape[0], size=n_batch)
        # print(features[random_indices])
        session.run([rbm.update_w, rbm.update_b_h, rbm.update_b_v,
                     #rbm2.update_w, rbm2.update_b_h, rbm2.update_b_v,],
                     ],
                    feed_dict={input_placeholder: features[random_indices]})
        if i % 100 == 0:
            hidden = session.run(rbm.hidden_state_plus, feed_dict={input_placeholder: features})
            svc.fit(hidden[:-100], target[:-100])
            print('svc score with    rbm: {}, without rbm: {}'.format(
                svc.score(hidden[-100:], target[-100:]), direct_svc_score))
            print('reconstruction error: {}'.format(session.run(
                rbm.reconstruction_error, feed_dict={input_placeholder: features[random_indices]})))
            hd = session.run(rbm.hidden_state_plus, feed_dict={input_placeholder: features[random_indices]})
            p = session.run(rbm.state_probabilities, feed_dict={input_placeholder: features[random_indices],
                                                               rbm.hidden_placeholder: hd})
            print(p)
    for i in range(n_iterations):
        random_indices = np.random.randint(0, features.shape[0], size=n_batch)
        # print(features[random_indices])
        #hiddens = session.run(rbm.hidden_state_plus, feed_dict={input_placeholder: features[random_indices]})
        session.run([rbm2.update_w, rbm2.update_b_h, rbm2.update_b_v],
                    feed_dict={input_placeholder: features[random_indices]})
    for i in range(n_iterations):
        random_indices = np.random.randint(0, features.shape[0], size=n_batch)
        # print(features[random_indices])
        #hiddens = session.run(rbm2.hidden_state_plus, feed_dict={input_placeholder: features[random_indices]})
        session.run([rbm3.update_w, rbm3.update_b_h, rbm3.update_b_v],
                    feed_dict={input_placeholder: features[random_indices]})
        if i % 500 == 0:
            hidden = session.run(rbm3.hidden_state_plus, feed_dict={input_placeholder: features})
            svc.fit(hidden[:-100], target[:-100])
            print('svc score with    rbm: {}, without rbm: {}'.format(svc.score(hidden[-100:], target[-100:]), direct_svc_score))

        # print(features)
        # print(rbm.visible.eval(feed_dict={input_placeholder: features}))
        # input()
    import matplotlib.pyplot as plt
    for i in range(20):
        hidden = session.run(rbm2.hidden_state_plus, feed_dict={input_placeholder: features[0].reshape((1, 64))})
        hidden = np.random.randint(0, 2, size=(1, n_hidden3))
        hidden3 = session.run(rbm3.visible_from_hidden, feed_dict={rbm3.hidden_placeholder: hidden})
        hidden2 = session.run(rbm2.visible_from_hidden, feed_dict={rbm2.hidden_placeholder: hidden3})
        visible = session.run(rbm.visible_from_hidden, feed_dict={rbm.hidden_placeholder: hidden2})
        plt.imshow(visible.reshape(8,8), cmap='gray')
        plt.show()
    # print(hidden)
