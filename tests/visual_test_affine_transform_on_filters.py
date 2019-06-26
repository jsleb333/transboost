import matplotlib.pyplot as plt
import numpy as np
import torch

from transboost.aggregation_mechanism import TransformInvariantFeatureAggregation
from transboost.transboost_v2 import *
from transboost.datasets import get_train_valid_test_bank
from transboost.utils import FiltersGenerator

(Xtr, Ytr), (Xts, Yts), _, filter_bank = get_train_valid_test_bank()

fg = FiltersGenerator(filter_bank, margin=4, rotation=10, scale=.1, shear=10, n_transforms=5)
bank = fg.draw_n_examples_from_bank(3)
filters = fg.generate_filters(bank)

tifa = TransformInvariantFeatureAggregation(maxpool_shape=(-1,-1))

transformed_weights = []
for weights, pos, ats in zip(filters.weights, filters.pos, filters.affine_transforms):
    pad = tifa._compute_padding(weights)

    transformed_weights.append(tifa._transform_weights(weights, ats, pad))
    # transformed_weights.shape: (n_transforms, n_ch, filter_height+pad, filter_width+pad)


fig, axes = plt.subplots(3, 5)

for i, f in enumerate(transformed_weights):
    for j, w in enumerate(f):
        axes[i,j].imshow(w[0], cmap='RdBu_r')

plt.show()
