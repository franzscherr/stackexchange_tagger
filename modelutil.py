#!/usr/bin/python
# __________________________________________________________________________________________________
# Tensorflow model utils
#

import tensorflow as tf
import numpy as np


def weight_variable(shape, name, initial_value=None):
    """
    Obtain a weight matrix with given shape and given name. If no initial value is given use a predefined random rule.

    :param shape: The shape that the obtained weight matrix should have
    :param name: The tensorflow name to use for this variable
    :param initial_value: If not None, use the value of this for initialization
    :return: The created tensorflow variable
    """
    if initial_value is None:
        uniform_max = np.sqrt(6 / np.sum(shape))
        uniform_min = -uniform_max
        initial_value = tf.random_uniform(shape, uniform_min, uniform_max)
    return tf.Variable(initial_value=initial_value, name=name)


def bias_variable(shape, name, initial_value=None):
    """
    Obtain a bias variable with given shape and given name. If no initial value is given use 0.1

    :param shape: The shape that the obtained bias variable should have
    :param name: The tensorflow name to use for this variable
    :param initial_value: If not None, use the value of this for initialization
    :return: The created tensorflow variable
    """
    if initial_value is None:
        initial_value = .1
    return tf.Variable(initial_value=tf.constant(initial_value, shape=shape), name=name)
