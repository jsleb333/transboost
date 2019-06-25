import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from transboost.label_encoder import OneHotEncoder

from transboost.datasets import MNISTDataset, get_train_valid_test_bank
from transboost.transboost_v2 import TransBoost, advance_to_the_next_layer, \
    get_multi_layers_random_features, get_multi_layers_filters, WLRidge
from transboost.utils import make_fig_axes, FiltersGenerator


(Xtr, Ytr), (X_val, Y_val), (Xts, Yts), filter_bank = get_train_valid_test_bank(
    dataset='mnist',
    valid=200,
    center=True,
    reduce=True,
    shuffle=101,
    n_examples=1000,
    bank_ratio=0.05,
    device='cuda'
)

amount_of_transformations = list(range(5, 100, 5))
train_accuracies = list()
val_accuracies = list()
encoder = OneHotEncoder(Ytr)
weak_learner = WLRidge
encoded_Y, weights = encoder.encode_labels(Ytr)
np.random.seed(101)
for nt in amount_of_transformations:
    print(f'Beginning calculations for {nt} transformations')
    temp_tr = list()
    temp_val = list()
    for i in range(10):
        print(f'iteration: {i}')
        # generate filters
        filters_generator = FiltersGenerator(filter_bank, filters_shape=(5, 5), rotation=15, scale=.1, shear=15, n_transforms=nt, margin=2)
        filters = get_multi_layers_filters(filters_generator, [100])
        # generate attribute and train weak learner
        S_tr = get_multi_layers_random_features(Xtr, filters)
        weak_predictor = weak_learner().fit(S_tr, encoded_Y, weights)
        # calculate train accuracy
        weak_prediction = weak_predictor.predict(S_tr)
        Y_pred = encoder.decode_labels(weak_prediction)
        train_acc = accuracy_score(y_true=Ytr, y_pred=Y_pred)
        temp_tr.append(train_acc)
        # calculate validation accuracy
        S_val = get_multi_layers_random_features(X_val, filters)
        weak_prediction_val = weak_predictor.predict(S_val)
        Y_val_pred = encoder.decode_labels(weak_prediction_val)
        val_acc = accuracy_score(y_true=Y_val, y_pred=Y_val_pred)
        temp_val.append(val_acc)
    train_accuracies.append(np.mean(temp_tr))
    val_accuracies.append(np.mean(temp_val))

# plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(amount_of_transformations, train_accuracies,  color='tab:blue')
ax.plot(amount_of_transformations, val_accuracies, color='tab:orange')
plt.show()
