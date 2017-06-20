import numpy as np
import matplotlib.pyplot as plt

from vectorspacemodel import StackExchangeVectorSpace


# directory = '/run/media/scherr/Daten/kddm/askubuntu.com'
# directory = '/run/media/scherr/Daten/kddm/android.stackexchange.com'
directory = './chess.stackexchange.com'
vsm = StackExchangeVectorSpace(directory_path=directory, significance_limit_min=100, significance_limit_max_factor=.75)

plt.hist(np.sum(vsm.target, axis=-1))
plt.show()