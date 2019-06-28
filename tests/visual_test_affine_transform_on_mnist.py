import matplotlib.pyplot as plt
import numpy as np
import torch

from transboost.datasets import MNISTDataset
from transboost.aggregation_mechanism import AffineTransform, RandomAffineSampler
from transboost.datasets import get_train_valid_test_bank
from transboost.utils import make_fig_axes

(Xtr, Ytr), (Xts, Yts), _, filter_bank = get_train_valid_test_bank(center=True, reduce=True)

N = 24
fig, ax = make_fig_axes(N)

# ats = [AffineTransform(rotation=3*(i-N/2)/360*2*np.pi, center=(14,14)) for i in range(N)]
rats = RandomAffineSampler(rotation=15, scale_x=.1, scale_y=.1, shear_x=15, shear_y=15, angle_unit='degrees')
ats = [rats.sample_transformation(center=(14,14)) for _ in range(N)]

x = Xtr[0,0]

for i, at in enumerate(ats):
    ax[i].imshow(at(x))
    ax[i].set_title(f'{at.rotation*360/2/np.pi:.2f}')

plt.show()
