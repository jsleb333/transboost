from sacred import Experiment
from sacred.observers import MongoObserver
from main_transboost_v2 import main
from transboost.callbacks.sacred_callbacks import SacredMetricsCallback
from transboost.datasets import MNISTDataset, CIFAR10Dataset

ex = Experiment()

ex.observers.append(MongoObserver.create(url='mongodb://mongo_user:mongo_password@127.0.0.1:27017',
                                         db_name='sacred'))


@ex.config
def my_config():
    m = 60000
    val = 6000
    dataset = 'mnist'
    center = True
    reduce = True
    encodings = 'onehot'
    wl = 'ridge'
    fs = 11
    fsh = 0
    n_layers = 2
    n_filters_per_layer = [50]
    bank_ratio = .05
    fn = 'c'
    loc = 3
    rot = 4
    scale = 0
    shear = 4
    margin = 2
    nt = 40
    nl = 'maxpool'
    maxpool = -1
    max_round = 1000
    patience = 1000
    resume = 0
    device = 'cuda'
    seed = 101


@ex.automain
def launch_experiment(m, val, dataset, center, reduce, encodings, wl, fs, fsh, n_layers,
                      n_filters_per_layer, bank_ratio, fn, loc, rot, scale, shear, margin, nt, nl,
                      maxpool, max_round, patience, resume, device, seed, _run):

    smc = SacredMetricsCallback(_run)
    main(m=m, val=val, dataset=dataset, center=center, reduce=reduce, encodings=encodings, wl=wl,
         fs=fs, fsh=fsh, n_layers=n_layers, n_filters_per_layer=n_filters_per_layer,
         bank_ratio=bank_ratio, fn=fn, loc=loc, rot=rot, scale=scale, shear=shear, margin=margin,
         nt=nt, nl=nl, maxpool=maxpool, max_round=max_round, patience=patience, resume=resume,
         device=device, seed=seed, smc=smc, run_info=_run)


