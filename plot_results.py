import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

directory = 'default_tagger_model'

with open(os.path.join(directory, 'train_log.npz'), 'rb') as f:
    npz = np.load(f)
    training_ce_errors = npz['training_cross_entropy_error']
    validation_ce_errors = npz['validation_cross_entropy_errors']
    validation_f1_scores = npz['validation_f1_scores']
    training_f1_scores = npz['training_f1_scores']

with open(os.path.join(directory, 'dbn_reconstruction_errors.pkl'), 'rb') as f:
    reconstruction_errors = pickle.load(f)

fig, axes = plt.subplots(2)
ax = axes[0]
ax.plot(training_ce_errors[2:])
ax.plot(validation_ce_errors[2:])
ax.grid()
ax.set_xlabel('Iteration')
ax.set_ylabel('Cross Entropy Error')
ax.legend(['Training set', 'Validation set'])

ax = axes[1]
ax.plot(training_f1_scores)
ax.plot(validation_f1_scores)
ax.grid()
ax.set_xlabel('Iteration')
ax.set_ylabel('F1 Score')
ax.legend(['Training set', 'Validation set'])
plt.show()

n_dbn_layers = len(reconstruction_errors)
fig, axes = plt.subplots(n_dbn_layers)
for i, ax in enumerate(axes):
    ax.plot(reconstruction_errors[i])
    ax.grid()
    ax.set_title('Deep Belief Layer {}'.format(i))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Reconstruction Error')
plt.show()
