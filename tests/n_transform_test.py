import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from transboost.label_encoder import OneHotEncoder

from transboost.datasets import MNISTDataset, get_train_valid_test_bank
from transboost.transboost_v2 import TransBoost, advance_to_the_next_layer, \
    MultiLayersRandomFeatures, get_multi_layers_filters, WLRidge
from transboost.utils import make_fig_axes, FiltersGenerator
from graal_utils import Timer


(Xtr, Ytr), (X_val, Y_val), (Xts, Yts), filter_bank = get_train_valid_test_bank(
    dataset='mnist',
    valid=200,
    center=True,
    reduce=True,
    shuffle=101,
    n_examples=1000,
    bank_ratio=0.05,
    # device='cuda'
)

n_transformations = np.arange(5, 11, 5)
train_accuracies = []
val_accuracies = []
encoder = OneHotEncoder(Ytr)
weak_learner = WLRidge(encoder=encoder)
# encoded_Y, weights = encoder.encode_labels(Ytr)
np.random.seed(101)
n_it = 3
for nt in n_transformations:
    with Timer():
        print(f'Beginning calculations for {nt} transformations')
        temp_tr = []
        temp_val = []
        for i in range(n_it):
            print(f'iteration: {i}')
            # generate filters
            filters_generator = FiltersGenerator(filter_bank, filters_shape=(5, 5), rotation=15, scale=.1, shear=15, n_transforms=nt, margin=2)
            filters = get_multi_layers_filters(filters_generator, [100])
            # generate attribute and train weak learner

            aggregation_mechanism = MultiLayersRandomFeatures(locality=3, maxpool_shape=(-1,-1,-1))
            S_tr = aggregation_mechanism(Xtr, filters)

            weak_predictor = weak_learner().fit(S_tr, Ytr)

            # calculate training accuracy
            train_acc = weak_predictor.evaluate(S_tr, Ytr)
            temp_tr.append(train_acc)
            # calculate validation accuracy
            S_val = aggregation_mechanism(X_val, filters)
            val_acc = weak_predictor.evaluate(S_val, Y_val)
            temp_val.append(val_acc)
        train_accuracies.append(np.mean(temp_tr))
        val_accuracies.append(np.mean(temp_val))

# plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(n_transformations, train_accuracies,  color='tab:blue')
ax.plot(n_transformations, val_accuracies, color='tab:orange')
plt.show()
