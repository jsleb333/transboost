import torch
import logging

from transboost.transboost_v2 import TransBoost
from transboost.label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder
from transboost.weak_learner import *
from transboost.callbacks import *
from transboost.datasets import get_train_valid_test_bank, MNISTDataset, CIFAR10Dataset
from transboost.utils import parse, FiltersGenerator, Filters
from graal_utils import timed


@timed
@parse
def main(m=60_000, val=10_000, dataset='mnist', center=True, reduce=True,
         encodings='onehot', wl='ridge',
         fs=5, fsh=0, n_layers=1, n_filters_per_layer=[10],
         bank_ratio=.05, fn='c',
         loc=3, rot=0, scale=.0, shear=0, margin=2, nt=1,
         nl='maxpool', maxpool=-1,
         max_round=1000, patience=1000, resume=0,
         device='cpu', seed=101,
         ):

    # Seed
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Data preparation
    (Xtr, Ytr), (X_val, Y_val), (Xts, Yts), filter_bank = get_train_valid_test_bank(
        dataset=dataset,
        valid=val,
        center=center,
        reduce=center,
        shuffle=seed,
        n_examples=m,
        bank_ratio=bank_ratio,
        device=device
    )

    # Choice of encoder
    if encodings == 'onehot':
        encoder = OneHotEncoder(Ytr)
    elif encodings == 'allpairs':
        encoder = AllPairsEncoder(Ytr)
    else:
        encoder = LabelEncoder.load_encodings(encodings)
        if all(label.isdigit() for label in encoder.labels_encoding):
            encoder = LabelEncoder({int(label):encoding for label, encoding in encoder.labels_encoding.items()})
    logging.info(f'Encoding: {encodings}')

    filename = f'transboost-d={dataset}-e={encodings}-wl={wl}'

    # Choice of weak learner
    kwargs = {}

    if wl == 'ridge':
        # Preparing filters
        # n_filters_per_layer = [int(f) for f in n_filters_per_layer]
        if len(n_filters_per_layer) == 1:
            n_filters_per_layer = n_filters_per_layer*n_layers

        filename += f'-nfperlayer={n_filters_per_layer}'
        filename += f'-fs={fs}'
        if fsh:
            filename += f'_to_{fsh}'
            fs = (fs, fhs)
        else:
            fs = (fs, fs)
        if loc != -1: filename += f'-loc={loc}'

        activation = None
        if 'maxpool' in nl:
            filename += f'-maxpool{maxpool}'
        if 'relu' in nl:
            filename += f'-relu'
            activation = torch.nn.functional.relu
        elif 'sigmoid' in nl:
            filename += f'-sigmoid'
            activation = torch.sigmoid

        filename += '-bank'

        if fn:
            filename += f'_{fn}'

        f_proc = []
        if 'c' in fn:
            f_proc.append(center_weight)
        if 'n' in fn:
            f_proc.append(normalize_weight)
        if 'r' in fn:
            f_proc.append(reduce_weight)

        filters_generator = FiltersGenerator(filter_bank, filters_shape=fs, rotation=rot, scale=scale, shear=shear, n_transforms=nt, margin=margin)
        weak_learner = WLRidge

    else:
        raise ValueError(f'Invalid weak learner name: "{wl}".')

    logging.info(f'Weak learner: {type(weak_learner).__name__}')


    ### Callbacks
    ckpt = ModelCheckpoint(filename=filename+'-{round}.ckpt', dirname='./results')
    logger = CSVLogger(filename=filename+'-log.csv', dirname='./results/log')
    zero_risk = BreakOnZeroRiskCallback()
    callbacks = [ckpt,
                logger,
                zero_risk,
                ]

    logging.info(f'Filename: {filename}')

    ### Fitting the model
    if not resume:
        logging.info(f'Beginning fit with filters per layers={n_filters_per_layer} and patience={patience}.')
        qb = TransBoost(filters_generator,
                        weak_learner,
                        encoder=encoder,
                        patience=patience,
                        n_filters_per_layer=n_filters_per_layer,
                        n_layers=n_layers,
                        callbacks=callbacks)
        qb.fit(Xtr, Ytr, X_val=X_val, Y_val=Y_val, **kwargs)
    ### Or resume fitting a model
    else:
        logging.info(f'Resuming fit with max_round_number={max_round}.')
        qb = TransBoost.load(f'results/{filename}-{resume}.ckpt')
        qb.resume_fit(Xtr, Ytr,
                      X_val=X_val, Y_val=Y_val,
                      max_round_number=max_round,
                      **kwargs)
    print(f'Best round recap:\nBoosting round {qb.best_round.step_number+1:03d} | Train acc: {qb.best_round.train_acc:.3%} | Valid acc: {qb.best_round.valid_acc:.3%} | Risk: {qb.best_round.risk:.3f}')
    if val:
        print(f'Test accuracy on best model: {qb.evaluate(Xts, Yts):.3%}')
        print(f'Test accuracy on last model: {qb.evaluate(Xts, Yts, mode="last"):.3%}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, style='{', format='[{levelname}] {message}')
    main(m=100, val=10)
