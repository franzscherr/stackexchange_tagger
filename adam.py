#!/usr/bin/python
# __________________________________________________________________________________________________
# Custom implementation of ADAM optimizer as a helper
#

import tensorflow as tf


class ADAMOptimizer(object):
    def __init__(self, parameters, minimize=True, learning_rate=.001, beta1=.9, beta2=.99, epsilon=1e-8):
        """

        :param parameters:
        :param minimize:
        :param learning_rate:
        :param beta1:
        :param beta2:
        :param epsilon:
        """
        self.parameters = parameters
        self.minimize = minimize
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        with tf.name_scope('ADAMOptimizer'):
            self.first_moment = tf.Variable(initial_value=tf.zeros(parameters.get_shape(), dtype=tf.float32),
                                            trainable=False)
            self.second_moment = tf.Variable(initial_value=tf.zeros(parameters.get_shape(), dtype=tf.float32),
                                             trainable=False)

    def train_step(self, gradient):
        """

        :param gradient:
        :return:
        """
        with tf.name_scope('ADAMOptimizer'):
            first_moment = tf.assign(self.first_moment, (self.beta1 * self.first_moment +
                                                         (1 - self.beta1) * gradient) / (1 - self.beta1))
            second_moment = tf.assign(self.second_moment, (self.beta2 * self.second_moment +
                                                           (1 - self.beta2) * gradient**2) / (1 - self.beta2))
            if self.minimize:
                return tf.assign_add(self.parameters,
                                     - self.learning_rate * first_moment / (tf.sqrt(second_moment) + self.epsilon))
            else:
                return tf.assign_add(self.parameters,
                                     self.learning_rate * first_moment / (tf.sqrt(second_moment) + self.epsilon))
