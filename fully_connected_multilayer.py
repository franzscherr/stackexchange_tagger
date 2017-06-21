#!/usr/bin/python
# __________________________________________________________________________________________________
# Fully Connected Multilayer Network
#

import tensorflow as tf

import modelutil
from dbn import DBN


class FullyConnectedMultilayer(object):
    def __init__(self, layer_sizes, input_tensor, activation=tf.nn.elu, keep_probability_placeholder=None,
                 last_part=True, name='fully_connected', initial_weight_list=None, initial_bias_list=None,
                 batch_normalize=False, stop_gradient_after_layer=None, initial_batch_normalization_parameters=None):
        """
        Implements a fully connected multilayer network.
        
        :param layer_sizes: The layer sizes in the network as a list
        :param input_tensor: The input tensor to feed into the network
        :param activation: The activation function, either an appropriate tensorflow function or a list with same size
                           as the layer_sizes of activation functions.
        :param keep_probability_placeholder: The dropout placeholder to use as keep probability
        :param last_part: Boolean indicating if the activation function should not be applied to the last part, i.e.
                          this network part is the last part of the network.
        :param name: The name to use for this network part
        :param initial_weight_list: A list of weights that are sequentially used to initialize the weights, if None
                                    initialize random. Each element of the list is a weight matrix: n_input x n_output
        :param initial_bias_list: A list of biases that are sequentially used to init the biases, if None -> random
                                  Each element of the list should be a one dimensional vector holding the biases.
        :param batch_normalize: Use batch normalization
        :param stop_gradient_after_layer: If given apply a stop_gradient operation after given layer
        :param initial_batch_normalization_parameters: A list for each layer containing tuples of the biases and
                                                       variances: [(alpha, beta), ...]. If None -> random
        """
        shape = input_tensor.get_shape().as_list()

        input_size = 1
        for dim_size in shape[1:]:
            input_size *= dim_size
        
        current_layer = tf.reshape(input_tensor, [-1, input_size])
        current_layer_size = input_size

        self.weights = []
        self.biases = []
        self.batch_normalize = batch_normalize

        # inner layers
        with tf.name_scope(name=name):
            for i in range(len(layer_sizes)):
                with tf.name_scope(name='layer_{}'.format(i)):
                    layer_size = layer_sizes[i]

                    # set up weights
                    initial_weight = None
                    if initial_weight_list is not None:
                        initial_weight = initial_weight_list[i]
                    initial_bias = None
                    if initial_bias_list is not None:
                        initial_bias = initial_bias_list[i]
                    W = modelutil.weight_variable([current_layer_size, layer_size], 'weights',
                                                  initial_value=initial_weight)
                    b = modelutil.bias_variable([layer_size], 'bias', initial_value=initial_bias)
                    self.weights.append(W)
                    self.biases.append(b)
                    h = tf.matmul(current_layer, W) + b
                    if keep_probability_placeholder is not None:
                        h = tf.nn.dropout(h, keep_probability_placeholder)

                    # activation (not for output)
                    if not last_part or i != len(layer_sizes) - 1:
                        if type(activation) is list:
                            h = activation[i](h, name='activation')
                        else:
                            h = activation(h, name='activation')

                    current_layer_size = layer_size
                    current_layer = h
                    if stop_gradient_after_layer is not None and stop_gradient_after_layer == i:
                        current_layer = tf.stop_gradient(current_layer, name='stop_gradient')
        self.output = current_layer

    def parameter_norm(self, norm_type='l2'):
        """
        returns a tensorflow node representing the norm of all network parameters
        
        :param norm_type: Which norm to apply, typically 'l2'
        :return: A tensorflow node representing the total parameter norm
        """
        if norm_type == 'l2':
            return sum(map(tf.nn.l2_loss, self.weights))
        else:
            raise NotImplementedError('Currently other norm than l2')

    def transfer_from_dbn(self, session, dbn):
        for i, t in enumerate(zip(self.weights, self.biases)):
            if len(dbn.machines) <= i:
                break
            w, b = t
            assign_weight_op = tf.assign(w, dbn.machines[i].w)
            assign_bias_op = tf.assign(b, dbn.machines[i].b_h)
            session.run([assign_bias_op, assign_weight_op])


if __name__ == '__main__':
    inp = tf.constant(1.0, shape=(1, 5, 5, 3))
    fcp = FullyConnectedMultilayer([10, 10, 5], inp, last_part=True)
    q = fcp.output

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(q))
