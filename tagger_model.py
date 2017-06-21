#!/usr/bin/python
# __________________________________________________________________________________________________
# Tagging model
#

import os
import sys
import tensorflow as tf
import numpy as np
from scipy import sparse, linalg
from sklearn.model_selection import train_test_split
import logging
import time
import pickle
import yaml

from vectorspacemodel import StackExchangeVectorSpace
from fully_connected_multilayer import FullyConnectedMultilayer
from dbn import DBN
from early_stopping import EarlyStopping


handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logging.root.addHandler(handler)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# directory = 'chess.stackexchange.com'
# directory = '/run/media/scherr/Daten/kddm/askubuntu.com'
directory = '/run/media/scherr/Daten/kddm/android.stackexchange.com'
vsm = StackExchangeVectorSpace(directory_path=directory, significance_limit_min=100, significance_limit_max_factor=.75)

# project_dir = time.strftime('tagger_model_%Y-%m-%d_%H-%M-%S')
project_dir = 'default_tagger_model'

print('number of supervised samples: {}'.format(vsm.supervised_data.shape[0]))
print('number of terms: {:6d}'.format(vsm.supervised_data.shape[-1]))
print('number of tags: {:6d}'.format(vsm.target.shape[-1]))
print('----------------------')
print('Project directory is: {}'.format(project_dir))

# __________________________________________________________________________________________________
# Accept only tags that are at least 200 in count
tag_sum = np.array(vsm.target.sum(axis=0))[0]
sorted_indices = np.argsort(tag_sum)[::-1]
tag_sum_sorted = tag_sum[sorted_indices]
n_tags = np.sum(tag_sum > 10)
vsm.target = vsm.target[:, sorted_indices[:n_tags]]
print('New tag count: {}'.format(n_tags))

n_documents = vsm.supervised_data.shape[0]
n_input = vsm.supervised_data.shape[1]
n_tags = vsm.target.shape[1]
test_only = False
p = dict()
if not test_only:
    p['n_iterations'] = 3000
    p['dbn_iterations'] = [1000, 1000]
else:
    p['n_iterations'] = 0
    p['dbn_iterations'] = [0, 0]
p['learning_rate'] = 1e-2
p['learning_rate_end'] = 1e-3
p['batch_size'] = 2000
p['test_batch_size_max'] = 1000
p['parameter_penalty_contribution'] = 0
p['layer_sizes'] = [1000, 600, 1000, n_tags]
p['dbn_layer_count'] = 2
p['use_lsa'] = False
p['use_dbn'] = True
p['lsa_components'] = 600
p['dbn_batch_size'] = 100

config_file_name = 'config.yaml'
load = False
if os.path.exists(os.path.join(project_dir, config_file_name)) and load:
    with open(os.path.join(project_dir, config_file_name), 'r') as f:
        p = yaml.load(f.read())
        logger.info('restored config from {}'.format(f.name))
else:
    os.makedirs(project_dir, exist_ok=True)
    with open(os.path.join(project_dir, config_file_name), 'w') as f:
        yaml.dump(p, f, default_flow_style=False)

n_iterations = p['n_iterations']
dbn_iterations = p['dbn_iterations']
learning_rate = p['learning_rate']
learning_rate_end = p['learning_rate_end']
batch_size = p['batch_size']
test_batch_size_max = p['test_batch_size_max']
parameter_penalty_contribution = p['parameter_penalty_contribution']
layer_sizes = p['layer_sizes']
dbn_layer_count = p['dbn_layer_count']
use_lsa = p['use_lsa']
use_dbn = p['use_dbn']
lsa_components = p['lsa_components']
dbn_batch_size = p['dbn_batch_size']

# __________________________________________________________________________________________________
# Apply latent semantic analysis
if use_lsa:
    n_input = lsa_components
    _, _, vectors = sparse.linalg.svds(vsm.data, n_input)
    vsm.supervised_data = vsm.supervised_data.dot(vectors.T)
    del vsm.data
    del vsm.unsupervised_data

# __________________________________________________________________________________________________
# Gradient descent very sensible to ratio of tags given in batch, hence sparse data like these
# tags pose a problem
# vsm.target = vsm.target[:, 1726]
# t = vsm.target.todense()
# indices = np.where(t == 1)[0]
# print(indices)
# not_indices = np.where(t == 0)[0]
# ind = np.hstack((indices, not_indices[:indices.shape[0]]))
# vsm.target = vsm.target[ind, :]
# vsm.supervised_data = vsm.supervised_data[ind, :]
# n_tags = 1

train_data, test_data, train_target, test_target = train_test_split(vsm.supervised_data, vsm.target, test_size=0.2)
train_data, validation_data, train_target, validation_target = train_test_split(train_data, train_target, test_size=0.2)
test_batch_size = test_target.shape[0] if test_target.shape[0] < test_batch_size_max else test_batch_size_max

document_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, n_input), name='document_vector')
target_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, n_tags), name='tag_vector')
tag_mask = tf.placeholder(dtype=tf.float32, shape=(1, n_tags), name='tag_mask')

keep_probability_placeholder = tf.placeholder(dtype=tf.float32)

dbn = None
if use_dbn and len(layer_sizes) > dbn_layer_count:
    dbn = DBN(layer_sizes[:dbn_layer_count], document_placeholder, n_fantasy_states=dbn_batch_size, learning_rate=1e-2,
              persistent_contrastive_divergence=False, constrastive_divergence_level=1)

if use_dbn:
    activation_functions = [tf.nn.sigmoid] * (dbn_layer_count - 1) + [tf.nn.elu] * (len(layer_sizes) - dbn_layer_count)
    stop_after = dbn_layer_count - 1
    batch_normalize = [False] * (dbn_layer_count - 1) + [True] * (len(layer_sizes) - dbn_layer_count)
else:
    activation_functions = tf.nn.elu
    stop_after = None
    batch_normalize = True
network = FullyConnectedMultilayer(layer_sizes, document_placeholder, activation=activation_functions,
                                   stop_gradient_after_layer=stop_after, batch_normalize=batch_normalize,
                                   keep_probability_placeholder=keep_probability_placeholder)
error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=target_placeholder, logits=network.output) * tag_mask, name='error') + \
        parameter_penalty_contribution * network.parameter_norm()
        # 10 * tf.reduce_mean((tf.reduce_sum(tf.sigmoid(network.output), axis=-1) -
        #                     tf.reduce_sum(target_placeholder, axis=-1))**2)
prediction = tf.round(tf.sigmoid(network.output)) * tag_mask
true_positive = tf.reduce_sum(prediction * target_placeholder)
true_negative = tf.reduce_sum((1 - prediction) * (1 - target_placeholder))
false_positive = tf.reduce_sum(prediction * (1 - target_placeholder))
false_negative = tf.reduce_sum((1 - prediction) * target_placeholder)
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1_score = 2 * precision * recall / (precision + recall)

classification_rate = 1. - tf.reduce_mean(tf.abs(target_placeholder - tf.round(tf.sigmoid(network.output))))
learning_rate_placeholder = tf.placeholder(dtype=tf.float32)
train_step = tf.train.AdamOptimizer(learning_rate_placeholder).minimize(error)

save = False
saver = tf.train.Saver() if save else None
save_dir = os.path.join(project_dir, 'model_saved_state') if save else None
save_path = os.path.join(save_dir, 'model.cpkt') if save else None

early_stopping = EarlyStopping(n_consecutive_steps=100)


def create_equalized_batch(tag_ind):
    t_mask = np.zeros((1, n_tags))
    t_mask[0, tag_ind] = 1

    tag_target = train_target[:, tag_ind]
    if hasattr(tag_target, 'todense'):
        tag_target = tag_target.todense()
    indices = np.where(tag_target == 1)[0]
    not_indices = np.where(tag_target == 0)[0]
    if indices.shape[0] < 10:
        raise ValueError('Tag error')
    indices = np.random.choice(indices, size=int(np.ceil(batch_size/2)))
    not_indices = np.random.choice(not_indices, size=int(np.floor(batch_size/2)))
    indices = np.concatenate((indices, not_indices))
    t_batch = train_data[indices, :]
    if hasattr(t_batch, 'todense'):
        t_batch = t_batch.todense()
    t_target = train_target[indices, :]
    if hasattr(t_target, 'todense'):
        t_target = t_target.todense()

    return t_batch, t_target, t_mask

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if save:
        if os.path.exists(save_dir):
            saver.restore(sess, save_path=save_path)
            logger.info('-- model restored from {}'.format(save_path))
        else:
            os.makedirs(save_dir)

    # ______________________________________________________________________________________________
    # Training

    if use_dbn:
        logger.info('DBN start')
        reconstruction_errors = dbn.train(sess, dbn_iterations, vsm.unsupervised_data, saver=saver, save_path=save_path,
                                          batch_size=dbn_batch_size)
        with open(os.path.join(project_dir, 'dbn_reconstruction_errors.pkl'), 'wb') as f:
            pickle.dump(reconstruction_errors, f)
        network.transfer_from_dbn(sess, dbn)
        logger.info('DBN end')

    training_ce_errors = []
    validation_ce_errors = []
    validation_f1_scores = []
    tag_fault = np.ones((1, n_tags))
    if n_iterations != 0:
        learning_rate_decay = np.power(learning_rate_end / learning_rate, 1 / n_iterations)
    else:
        learning_rate_decay = 1
    for i in range(n_iterations):
        tag_index = np.random.randint(0, n_tags)
        if learning_rate > learning_rate_end:
            learning_rate *= learning_rate_decay

        # __________________________________________________________________________________________
        # Train step

        batch_indices = np.random.randint(0, train_data.shape[0], size=batch_size)
        batch = train_data[batch_indices]
        if hasattr(batch, 'todense'):
            batch = batch.todense()
        # batch[batch.nonzero()] = 1
        target = train_target[batch_indices].todense()
        if hasattr(target, 'todense'):
            target = target.todense()
        mask = np.ones((1, n_tags))
        # try:
        #     batch, target, mask = create_equalized_batch(tag_index)
        # except ValueError:
        #     tag_fault[:, tag_index] = 0
        #     print('Tag fault')
        #     continue
        feed_dict = dict()
        feed_dict[document_placeholder] = batch
        feed_dict[target_placeholder] = target
        feed_dict[tag_mask] = mask
        feed_dict[learning_rate_placeholder] = learning_rate
        feed_dict[keep_probability_placeholder] = .8
        _, ce_err = sess.run([train_step, error], feed_dict=feed_dict)

        # __________________________________________________________________________________________
        # Validation on validation set

        batch_indices = np.random.randint(0, validation_data.shape[0], size=batch_size)
        document_batch = validation_data[batch_indices]
        if hasattr(document_batch, 'todense'):
            document_batch = document_batch.todense()
        # document_batch[document_batch.nonzero()] = 1
        feed_dict[keep_probability_placeholder] = 1
        feed_dict[document_placeholder] = document_batch
        feed_dict[target_placeholder] = validation_target[batch_indices]
        if hasattr(feed_dict[target_placeholder], 'todense'):
            feed_dict[target_placeholder] = feed_dict[target_placeholder].todense()
        ce_validation_error, validation_f1_score = sess.run([error, f1_score], feed_dict=feed_dict)

        training_ce_errors.append(ce_err)
        validation_ce_errors.append(ce_validation_error)
        validation_f1_scores.append(validation_f1_score)

        # __________________________________________________________________________________________
        # Apply early stopping

        early_stopping.update(-validation_f1_score)

        # __________________________________________________________________________________________
        # Save and test miss-classification on test set

        if early_stopping.should_save() and save:
            saver.save(sess, save_path=save_path)
            logger.info('-- model saved to {}'.format(save_path))

        if i % 20 == 0:
            batch_indices = np.random.randint(0, validation_data.shape[0], size=test_batch_size)
            document_batch = validation_data[batch_indices]
            if hasattr(document_batch, 'todense'):
                document_batch = document_batch.todense()
            # document_batch[document_batch.nonzero()] = 1
            feed_dict[document_placeholder] = document_batch
            feed_dict[target_placeholder] = validation_target[batch_indices]
            if hasattr(feed_dict[target_placeholder], 'todense'):
                feed_dict[target_placeholder] = feed_dict[target_placeholder].todense()
            feed_dict[keep_probability_placeholder] = 1.
            # classification_rate_run = sess.run(classification_rate, feed_dict=feed_dict)
            f1_score_run = sess.run(f1_score, feed_dict=feed_dict)
            logger.info('train error in iteration {} of {} is {:.4f}'.format(i, n_iterations, ce_err))
            logger.info('f1 score of validation in iteration {} of {} is {:.4f}'.format(i, n_iterations, f1_score_run))

            with open(os.path.join(project_dir, 'train_log.npz'), 'wb') as f:
                np.savez(f, training_cross_entropy_error=np.array(training_ce_errors),
                         validation_ce_errors=np.array(validation_ce_errors),
                         validation_f1_scores=np.array(validation_f1_scores))
            batch_indices = np.random.randint(0, train_data.shape[0], size=test_batch_size)
            document_batch = train_data[batch_indices]
            if hasattr(document_batch, 'todense'):
                document_batch = document_batch.todense()
            feed_dict[document_placeholder] = document_batch
            feed_dict[target_placeholder] = train_target[batch_indices]
            if hasattr(feed_dict[target_placeholder], 'todense'):
                feed_dict[target_placeholder] = feed_dict[target_placeholder].todense()
            f1_score_run = sess.run(f1_score, feed_dict=feed_dict)
            logger.info('f1 score of train in iteration {} of {} is {:.4f}'.format(i, n_iterations, f1_score_run))

        if early_stopping.should_abort():
            print('stop')

    # ______________________________________________________________________________________________
    # Testing

    logger.info('Testing starts')
    testing_samples = test_data.shape[0]
    if testing_samples > 100000:
        testing_samples = 100000
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(0, testing_samples, test_batch_size):
        high_i = i + test_batch_size
        if high_i > testing_samples:
            high_i = testing_samples
        batch = test_data[i:high_i]
        label = test_target[i:high_i]

        feed_dict = dict()
        feed_dict[document_placeholder] = batch
        if hasattr(feed_dict[document_placeholder], 'todense'):
            feed_dict[document_placeholder] = feed_dict[document_placeholder].todense()
        feed_dict[target_placeholder] = label
        if hasattr(feed_dict[target_placeholder], 'todense'):
            feed_dict[target_placeholder] = feed_dict[target_placeholder].todense()
        feed_dict[keep_probability_placeholder] = 1.
        feed_dict[tag_mask] = tag_fault
        temp_tp, temp_tn, temp_fp, temp_fn = \
            sess.run([true_positive, true_negative, false_positive, false_negative], feed_dict=feed_dict)
        tp += temp_tp
        tn += temp_tn
        fp += temp_fp
        fn += temp_fn
        print(tp, tn, fp, fn)
    print(tp, tn, fp, fn)
    final_precision = tp / (tp + fn)
    final_recall = tp / (tp + fp)
    final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall)
    print('f1 score: {:.5f}'.format(final_f1))
