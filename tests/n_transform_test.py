import logging

from sacred import Experiment
from sacred.observers import MongoObserver
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from transboost.label_encoder import OneHotEncoder

from transboost.datasets import MNISTDataset, get_train_valid_test_bank
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
    min_number_of_iterations = 25
    minim = 0
    maxim = 100
    step = 5
    criterion = 0.0001
    m = 10000
    val = 1000
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
    maxpool = (-1, -1, -1)
    device = 'cuda'
    seed = True


@ex.automain
def run(min_number_of_iterations, minim, maxim, step, criterion, m, val, dataset, center, reduce, fs, n_filters_per_layer, bank_ratio, loc, rot, scale, shear, margin, maxpool, device, seed, _run):
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
    filters_generator = FiltersGenerator(filter_bank, filters_shape=fs, rotation=rot, scale=scale,
                                         shear=shear, n_transforms=maxim, margin=margin)
    filters = get_multi_layers_filters(filters_generator, n_filters_per_layer)

    with open('n_transforms.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        aggregation_mechanism = MultiLayersRandomFeatures(loc, maxpool, None)
        amount_of_transformations = list(range(minim, maxim+step, step))
        train_accuracies = list()
        val_accuracies = list()
        weak_learner = WLRidge(encoder=encoder)
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
                    # generate filters
                    for f in filters:
                        affine_transforms = filters_generator._generate_affine_transforms(f.weights, f.pos, nt)
                        f.affine_transforms = affine_transforms
                    filters = get_multi_layers_filters(filters_generator, n_filters_per_layer)
                    # generate attribute and train weak learner
                    S_tr = aggregation_mechanism(Xtr, filters)
                    weak_predictor = weak_learner().fit(S_tr, Ytr)
                    # calculate train accuracy
                    weak_prediction = weak_predictor.predict(S_tr)
                    train_acc = weak_predictor.evaluate(S_tr, Ytr)
                    temp_tr.append(train_acc)
                    # calculate validation accuracy
                    S_val = aggregation_mechanism(X_val, filters)
                    weak_prediction = weak_predictor.predict(S_val)
                    val_acc = weak_predictor.evaluate(S_val, Y_val)
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
