import logging
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from transboost.label_encoder import OneHotEncoder

from transboost.datasets import MNISTDataset, get_train_valid_test_bank
from transboost.transboost_v2 import get_multi_layers_filters, WLRidge, MultiLayersRandomFeatures
from transboost.utils import  FiltersGenerator
from graal_utils import Timer

with open('n_transforms.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    aggregation_mechanism = MultiLayersRandomFeatures(3, (-1, -1), None)
    amount_of_transformations = list(range(5, 105, 5))
    train_accuracies = list()
    val_accuracies = list()
    weak_learner = WLRidge
    writer.writerow(['nt', 'train_acc', 'test_acc'])
    for nt in amount_of_transformations:
        with Timer():
            print(f'Beginning calculations for {nt} transformations')
            temp_tr = list()
            temp_val = list()
            delta = 1
            i = 0
            while delta > 0.0001:
                print(f'iteration: {i}')
                # prepare data
                (Xtr, Ytr), (X_val, Y_val), (Xts, Yts), filter_bank = get_train_valid_test_bank(
                    dataset='mnist',
                    valid=2000,
                    center=True,
                    reduce=True,
                    shuffle=True,
                    n_examples=10000,
                    bank_ratio=0.05,
                    device='cuda'
                )
                encoder = OneHotEncoder(Ytr)
                encoded_Y, weights = encoder.encode_labels(Ytr)
                # generate filters
                filters_generator = FiltersGenerator(filter_bank, filters_shape=(5, 5), rotation=15, scale=.1, shear=15, n_transforms=nt, margin=2)
                filters = get_multi_layers_filters(filters_generator, [100])
                # generate attribute and train weak learner
                S_tr = aggregation_mechanism(Xtr, filters)
                weak_predictor = weak_learner().fit(S_tr, encoded_Y, weights)
                # calculate train accuracy
                weak_prediction = weak_predictor.predict(S_tr)
                Y_pred = encoder.decode_labels(weak_prediction)
                train_acc = accuracy_score(y_true=Ytr, y_pred=Y_pred)
                temp_tr.append(train_acc)
                # calculate validation accuracy
                S_val = aggregation_mechanism(X_val, filters)
                weak_prediction_val = weak_predictor.predict(S_val)
                Y_val_pred = encoder.decode_labels(weak_prediction_val)
                val_acc = accuracy_score(y_true=Y_val, y_pred=Y_val_pred)
                temp_val.append(val_acc)
                if i >= 10:
                    delta = np.mean(temp_val) - val_acc
                i += 1
            t_acc = np.mean(temp_tr)
            v_acc = np.mean(temp_val)
            writer.writerow([nt, t_acc, v_acc])
            print(f'{nt}, {t_acc}, {v_acc}')
            train_accuracies.append(t_acc)
            val_accuracies.append(v_acc)

# plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(amount_of_transformations, train_accuracies,  color='tab:blue')
ax.plot(amount_of_transformations, val_accuracies, color='tab:orange')
plt.show()
