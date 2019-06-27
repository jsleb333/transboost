import torch
import numpy as np
import matplotlib.pyplot as plt

from transboost.datasets import MNISTDataset
from transboost.transboost_v2 import TransBoost, advance_to_the_next_layer, MultiLayersRandomFeatures, get_multi_layers_filters
from transboost.utils import make_fig_axes, FiltersGenerator, center_weight, reduce_weight
from sklearn.preprocessing import StandardScaler

Xtr, Ytr = MNISTDataset.load().get_train(center=False, reduce=False, shuffle=False)
Xtr, Ytr = MNISTDataset(Xtr[333:3333], Ytr[333:3333]).get_train(center=True, reduce=True, shuffle=False)

# Xtr = Xtr / 255
Xtr = torch.unsqueeze(torch.from_numpy(Xtr), dim=1)

# x, y = Xtr[333:340], Ytr[333:340]
x, y = Xtr, Ytr
eights = [i for i, yy in enumerate(y) if yy == 8]
print(eights)
x_idx = 0
n_filters = 5
filters_shape = (5,5)
n_layers = 4

col = 3 + (n_layers-1)*(n_filters+1)
fig, ax = make_fig_axes(n_filters*col, aspect_ratio=n_filters/col)

# ax = [[ax[i+j] if i < col else None for j in range(3)] for i in range(0, n_filters * col, n_filters)]
ax = [[ax[i*col+j] if i*col+j < n_filters*col else None for j in range(col)] for i in range(n_filters)]

for axs in ax:
    for a in axs:
        a.axis('off')

np.random.seed(42)
# f_proc = [center_weight, reduce_weight]
f_proc = []
filters_generator = FiltersGenerator(Xtr[-500:], margin=5, filters_shape=filters_shape, filters_preprocessing=f_proc)
mlf = get_multi_layers_filters(filters_generator, [n_filters]*n_layers)

# Layer 0
filters_layer_0 = mlf[0]

new_x = advance_to_the_next_layer(x, filters_layer_0)
# new_x = torch.from_numpy(StandardScaler().fit_transform(new_x.reshape(1000,-1)).reshape(new_x.shape))
new_x = torch.relu(new_x)
# new_x = torch.from_numpy(StandardScaler().fit_transform(new_x.reshape(1000,-1)).reshape(new_x.shape))

ax[(n_filters-1)//2][0].imshow(x[x_idx,0], cmap='RdBu_r')

for i in range(n_filters):
    ax[i][1].imshow(filters_layer_0.weights[i,0], cmap='RdBu_r')
    ax[i][2].imshow(new_x[x_idx,i], cmap='RdBu_r')

# Other layers
for l, filters in enumerate(mlf[1:]):
    new_x = advance_to_the_next_layer(new_x, filters)
    # new_x = torch.from_numpy(StandardScaler().fit_transform(new_x.reshape(1000,-1)).reshape(new_x.shape))
    new_x = torch.relu(new_x)
    # new_x = torch.from_numpy(StandardScaler().fit_transform(new_x.reshape(1000,-1)).reshape(new_x.shape))
    for i in range(n_filters):
        for j in range(n_filters):
            ax[j][(n_filters+1)*l+3+i].imshow(filters.weights[j,i], cmap='RdBu_r')

        ax[i][(n_filters+1)*l+3+n_filters].imshow(new_x[x_idx,i], cmap='RdBu_r')

plt.show()
