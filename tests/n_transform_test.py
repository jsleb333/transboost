import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import csv

from transboost.label_encoder import OneHotEncoder

from transboost.datasets import MNISTDataset, get_train_valid_test_bank
from transboost.transboost_v2 import TransBoost, advance_to_the_next_layer, \
    MultiLayersRandomFeatures, get_multi_layers_filters, WLRidge
from transboost.utils import make_fig_axes, FiltersGenerator
from graal_utils import Timer, timed

np.random.seed(101)

(Xtr, Ytr), (X_val, Y_val), (Xts, Yts), filter_bank = get_train_valid_test_bank(
    dataset='mnist',
    valid=10_000,
    center=True,
    reduce=True,
    shuffle=True,
    n_examples=11_000,
    bank_ratio=1/11,
    device='cpu'
)

n_transformations = np.arange(0, 151, 5)
n_it = 30
n_filters = 100
train_accuracies, val_accuracies = [], []
train_partial_means, val_partial_means = [], []
encoder = OneHotEncoder(Ytr)
weak_learner = WLRidge(encoder=encoder)
# encoded_Y, weights = encoder.encode_labels(Ytr)
# generate filters
filters_generator = FiltersGenerator(filter_bank, filters_shape=(5, 5), rotation=15, scale=.1, shear=15, n_transforms=max(n_transformations), margin=2)
filters = get_multi_layers_filters(filters_generator, [n_filters])

for nt in n_transformations:
    with Timer():
        print(f'Beginning calculations for {nt} transformations')
        temp_tr, temp_val = [], []
        for i in range(n_it):
            print(f'iteration: {i}')
            # generate attribute and train weak learner
            for f in filters:
                affine_transforms = filters_generator._generate_affine_transforms(f.weights, f.pos, nt)
                f.affine_transforms = affine_transforms

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
        train_accuracies.append(temp_tr)
        val_accuracies.append(temp_val)
        train_partial_means.append([np.mean(temp_tr[:i+1]) for i, _ in enumerate(temp_tr)])
        val_partial_means.append([np.mean(temp_val[:i+1]) for i, _ in enumerate(temp_val)])

# save the data
os.makedirs('./results/', exist_ok=True)
filename_train = f'./results/n_transforms_effect-train_accuracies-nt={n_transformations[0]}_to_{n_transformations[-1]}-n_it={n_it}-nf={n_filters}.csv'
with open(filename_train, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['nt'] + [it for it in range(n_it)])
    for i, nt in enumerate(n_transformations):
        csv_writer.writerow([nt] + train_accuracies[i])
os.makedirs('./results/', exist_ok=True)
filename_val = f'./results/n_transforms_effect-val_accuracies-nt={n_transformations[0]}_to_{n_transformations[-1]}-n_it={n_it}-nf={n_filters}.csv'
with open(filename_val, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([it for it in range(n_it)])
    for i, nt in enumerate(n_transformations):
        csv_writer.writerow([nt] + val_accuracies[i])

# # plot the data
# fig = plt.figure()
# ax = fig.subplots(len(n_transformations)+1, 1)
# ax_nt = ax[0]
# ax_nt.plot(n_transformations, [np.mean(acc) for acc in train_accuracies],  color='tab:blue')
# ax_nt.plot(n_transformations, [np.mean(acc) for acc in val_accuracies], color='tab:orange')
# for i, ax_partial in enumerate(ax[1:]):
#     ax_partial.plot(train_partial_means[i])
#     ax_partial.plot(val_partial_means[i])
# plt.show()
