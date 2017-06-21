#!/usr/bin/python
# __________________________________________________________________________________________________
# Early Stopping Helper
#

import numpy as np


class EarlyStopping(object):
    def __init__(self, n_consecutive_steps=5):
        """

        :param n_consecutive_steps:
        """
        self.n_consecutive_steps = n_consecutive_steps
        self.consecutive_steps = 0
        self.min_error = np.inf
        self.abort = False

    def update(self, error):
        if error < self.min_error:
            self.min_error = error
            self.consecutive_steps = 0
            self.abort = False
        else:
            self.consecutive_steps += 1
            if self.consecutive_steps >= self.n_consecutive_steps:
                self.abort = True

    def should_save(self):
        return self.consecutive_steps == 1

    def should_abort(self):
        return self.abort
