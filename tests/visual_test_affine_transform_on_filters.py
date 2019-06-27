import matplotlib.pyplot as plt
import numpy as np
import torch

from transboost.aggregation_mechanism import TransformInvariantFeatureAggregation
from transboost.transboost_v2 import *
from transboost.datasets import get_train_valid_test_bank
from transboost.utils import FiltersGenerator, center_weight, reduce_weight

(Xtr, Ytr), (Xts, Yts), _, filter_bank = get_train_valid_test_bank(center=True, reduce=True)

fg = FiltersGenerator(filter_bank, margin=4, rotation=15, scale=.1, shear=15, n_transforms=5, filters_preprocessing=[center_weight, reduce_weight])
bank = fg.draw_n_examples_from_bank(3)
filters = fg.generate_filters(bank)

tifa = TransformInvariantFeatureAggregation(maxpool_shape=(-1,-1))

transformed_weights = []
for weights, pos, ats in zip(filters.weights, filters.pos, filters.affine_transforms):
    pad = tifa._compute_padding(weights)

    transformed_weights.append(tifa._transform_weights(weights, ats, pad))
    # transformed_weights.shape: (n_transforms, n_ch, filter_height+pad, filter_width+pad)


v_max = torch.max(torch.cat(transformed_weights))
v_min = torch.min(torch.cat(transformed_weights))
vlim = max(v_max, -v_min)

fig, axes = plt.subplots(3, 5+2)

for i, f in enumerate(filters):
    w = f.weights
    axes[i,0].imshow(w[0], cmap='RdBu_r', vmax=vlim, vmin=-vlim)
    axes[i,1].imshow(np.pad(w[0].numpy(), pad, 'constant'), cmap='RdBu_r', vmax=vlim, vmin=-vlim)
    tw = transformed_weights[i]
    for j, w in enumerate(tw):
        axes[i,j+2].imshow(w[0], cmap='RdBu_r', vmax=vlim, vmin=-vlim)
        at = f.affine_transforms[j][0]
        axes[i,j+2].set_title(f'rot={at.rotation*360/2/np.pi:.1f} shear=({at.shear[0]*360/2/np.pi:.1f},{at.shear[1]*360/2/np.pi:.1f})')

plt.show()
