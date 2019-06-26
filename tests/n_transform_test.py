import logging

from sacred import Experiment
from sacred.observers import MongoObserver
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from transboost.label_encoder import OneHotEncoder

from transboost.datasets import MNISTDataset, get_train_valid_test_bank
<<<<<<< HEAD
from transboost.transboost_v2 import TransBoost, advance_to_the_next_layer, \
    MultiLayersRandomFeatures, get_multi_layers_filters, WLRidge
from transboost.utils import make_fig_axes, FiltersGenerator


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

amount_of_transformations = range(5, 10, 5)
train_accuracies = []
val_accuracies = []
encoder = OneHotEncoder(Ytr)
weak_learner = WLRidge(encoder)
# encoded_Y, weights = encoder.encode_labels(Ytr)
np.random.seed(101)
for nt in amount_of_transformations:
    print(f'Beginning calculations for {nt} transformations')
    temp_tr = []
    temp_val = []
    for i in range(10):
        print(f'iteration: {i}')
        # generate filters
        filters_generator = FiltersGenerator(filter_bank, filters_shape=(5, 5), rotation=15, scale=.1, shear=15, n_transforms=nt, margin=2)
        filters = get_multi_layers_filters(filters_generator, [100])
        # generate attribute and train weak learner

        aggregation_mechanism = MultiLayersRandomFeatures(locality=3, maxpool_shape=(-1,-1,-1))
        S_tr = aggregation_mechanism(Xtr, filters)

        weak_predictor = weak_learner().fit(S_tr, Ytr)
        # calculate train accuracy
        # weak_prediction = weak_predictor.predict(S_tr)
        # Y_pred = encoder.decode_labels(weak_prediction)
        train_acc = weak_predictor.evaluate(y_true=Ytr, y_pred=Y_pred)
        temp_tr.append(train_acc)
        # calculate validation accuracy
        # S_val = aggregation_mechanism(X_val, filters)
        # weak_prediction_val = weak_predictor.predict(S_val)
        # Y_val_pred = encoder.decode_labels(weak_prediction_val)
        # val_acc = accuracy_score(y_true=Y_val, y_pred=Y_val_pred)
        # temp_val.append(val_acc)
    train_accuracies.append(np.mean(temp_tr))
    # val_accuracies.append(np.mean(temp_val))

# plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot([amount_of_transformations], train_accuracies,  color='tab:blue')
# ax.plot([amount_of_transformations], val_accuracies, color='tab:orange')
plt.show()
=======
from transboost.transboost_v2 import get_multi_layers_filters, WLRidge, MultiLayersRandomFeatures
from transboost.utils import FiltersGenerator
from graal_utils import Timer

ex = Experiment('n_transforms test')

ex.observers.append(MongoObserver.create(url='mongodb://mongo_user:mongo_password@127.0.0.1:27017',
                                         db_name='sacred'))
logger = logging.getLogger()
logger.propagate = False
ex.logger = logger

@ex.config
def my_config():
    min_number_of_iterations = 50
    minim = 5
    maxim = 100
    step = 5
    criterion = 0.0001
    m = 60000
    val = 6000
    dataset = 'mnist'
    center = True
    reduce = True
    fs = (5,5)
    n_filters_per_layer = [100]
    bank_ratio = .05
    fn = 'c'
    loc = 3
    rot = 15
    scale = .1
    shear = 15
    margin = 2
    nt = 40
    maxpool = (-1, -1)
    device = 'cuda'
    seed = True


@ex.automain
def run(min_number_of_iterations, minim, maxim, step, criterion, m, val, dataset, center, reduce, fs, n_filters_per_layer, bank_ratio, loc, rot, scale, shear, margin, maxpool, device, seed, _run):
    with open('n_transforms.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        aggregation_mechanism = MultiLayersRandomFeatures(loc, maxpool, None)
        amount_of_transformations = list(range(minim, maxim+step, step))
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
                while delta > criterion:
                    print(f'iteration: {i}')
                    # prepare data
                    (Xtr, Ytr), (X_val, Y_val), (Xts, Yts), filter_bank = get_train_valid_test_bank(
                        dataset=dataset,
                        valid=val,
                        center=center,
                        reduce=reduce,
                        shuffle=seed,
                        n_examples=m,
                        bank_ratio=bank_ratio,
                        device=device
                    )
                    encoder = OneHotEncoder(Ytr)
                    encoded_Y, weights = encoder.encode_labels(Ytr)
                    # generate filters
                    filters_generator = FiltersGenerator(filter_bank, filters_shape=fs, rotation=rot, scale=scale, shear=shear, n_transforms=nt, margin=margin)
                    filters = get_multi_layers_filters(filters_generator, n_filters_per_layer)
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
                    if i >= min_number_of_iterations:
                        delta = np.mean(temp_val) - val_acc
                    i += 1
                t_acc = np.mean(temp_tr)
                v_acc = np.mean(temp_val)
                writer.writerow([nt, t_acc, v_acc])
                print(f'{nt}, {t_acc}, {v_acc}')
                _run.log_scalar('train accuracy', t_acc, nt)
                _run.log_scalar('validation accuracy', v_acc, nt)
                train_accuracies.append(t_acc)
                val_accuracies.append(v_acc)
    # plot the data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(amount_of_transformations, train_accuracies,  color='tab:blue')
    ax.plot(amount_of_transformations, val_accuracies, color='tab:orange')
    plt.show()

    _run.add_artifact('n_transforms.csv')



>>>>>>> 177a39e58977ba8b848d28fd628056d60d35adf3
