import torch
import numpy as np
import matplotlib.pyplot as plt

from transboost.datasets import MNISTDataset
from transboost.transboost_v2 import TransBoost, advance_to_the_next_layer, get_multi_layers_random_features
from transboost.utils import make_fig_axes, FiltersGenerator

Xtr, Ytr = MNISTDataset.load().get_train(center=True, reduce=True, shuffle=False)
Xtr = torch.unsqueeze(torch.from_numpy(Xtr), dim=1)

x, y = Xtr[333], Ytr[333]
n_filters = 3

filters_generator = FiltersGenerator(Xtr[55000:])
np.random.seed(42)
chosen_examples = filters_generator.draw_n_examples_from_bank(n_filters)
filters = filters_generator.generate_filters(chosen_examples)

new_x = advance_to_the_next_layer(torch.unsqueeze(x ,dim=0), filters)


N = 12
fig, ax = make_fig_axes(N, aspect_ratio=np.sqrt(N)/3)

ax[0].imshow(x[0], cmap='RdBu_r')
ax[3].imshow(new_x[0,0], cmap='RdBu_r')
ax[4].imshow(new_x[0,1], cmap='RdBu_r')
ax[5].imshow(new_x[0,2], cmap='RdBu_r')
ax[6].imshow(filters.weights[0,0], cmap='RdBu_r')
ax[7].imshow(filters.weights[1,0], cmap='RdBu_r')
ax[8].imshow(filters.weights[2,0], cmap='RdBu_r')
plt.show()
