import torch
import logging

from transboost.transboost import TransBoost
from transboost.label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder
from transboost.weak_learner import *
from transboost.callbacks import *
from transboost.datasets import MNISTDataset, CIFAR10Dataset
from transboost.utils import parse, timed


@timed
@parse
def main(m=60_000, val=10_000, da=0, dataset='mnist', center=True, reduce=True, encodings='onehot', wl='rccsparseridge', n_layers=3, n_filters_per_layer=[100], top_k=5, patience=1000, resume=0, n_filters=100, fs=5, fsh=0, locality=4, init_filters='from_bank', bank_ratio=.05, fn='c', seed=101, nl='maxpool', maxpool=8, device='cpu', margin=2):
    print(n_filters_per_layer)
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    ### Data loading
    if 'mnist' in dataset:
        data = MNISTDataset.load(dataset+'.pkl')
    elif 'cifar' in dataset:
        data = CIFAR10Dataset.load(dataset+'.pkl')

    (Xtr, Ytr), (X_val, Y_val), (Xts, Yts) = data.get_train_valid_test(valid=val, center=False, reduce=False, shuffle=seed)
    Xtr, Ytr = Xtr[:m], Ytr[:m]
    if da:
        logging.info(f'Adding {da} examples with data augmentation.')
        Xtr, Ytr = extend_mnist(Xtr, Ytr, N=da, degrees=degrees, scale=(1-scale, 1/(1-scale)), shear=shear)

    data.fit_scaler(Xtr, center=center, reduce=reduce)
    Xtr, Ytr = data.transform_data(Xtr.reshape(Xtr.shape[0],-1), Ytr)
    Xts, Yts = data.transform_data(Xts.reshape(Xts.shape[0],-1), Yts)
    if val:
        X_val, Y_val = data.transform_data(X_val.reshape(X_val.shape[0],-1), Y_val)
    else:
        X_val, Y_val = Xts, Yts

    Xtr, X_val, Xts = RandomConvolution.format_data(Xtr), RandomConvolution.format_data(X_val), RandomConvolution.format_data(Xts)

    logging.info('Boosting algorithm: TransBoost')
    logging.info(f'Loaded dataset: {dataset} (center: {center}, reduce: {reduce})')
    logging.info(f'Number of examples - train: {len(Xtr)}, valid: {len(X_val)}, test: {len(Xts)}')

    ### Choice of encoder
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

    ### Choice of weak learner
    kwargs = {}
    if wl in ['ds', 'decision-stump']:
        weak_learner = MulticlassDecisionStump()
        kwargs = dict(zip(('sorted_X', 'sorted_X_idx'), weak_learner.sort_data(Xtr)))
        kwargs['n_jobs'] = n_jobs

    elif wl in ['dt', 'decision-tree']:
        weak_learner = MulticlassDecisionTree(max_n_leaves=max_n_leaves)
        kwargs = dict(zip(('sorted_X', 'sorted_X_idx'), weak_learner.sort_data(Xtr)))
        kwargs['n_jobs'] = n_jobs
        filename += f'{max_n_leaves}'

    elif wl == 'ridge':
        weak_learner = WLThresholdedRidge(threshold=.5)

    elif wl.startswith('rcc') or wl.startswith('rlc'):
        if device.startswith('cuda'):
            Xtr = RandomConvolution.format_data(Xtr).to(device=device)
            X_val = RandomConvolution.format_data(X_val).to(device=device)
            Xts = RandomConvolution.format_data(Xts).to(device=device)

        filename += f'-nf={n_filters}-fs={fs}'
        if fsh: filename += f'_to_{fsh}'
        if wl.startswith('rlc'): filename += f'-loc={locality}'

        activation = None
        if 'maxpool' in nl:
            filename += f'-maxpool{maxpool}'
        if 'relu' in nl:
            filename += f'-relu'
            activation = torch.nn.functional.relu
        elif 'sigmoid' in nl:
            filename += f'-sigmoid'
            activation = torch.sigmoid

        filename += f'-{init_filters}'

        filter_bank = None
        if init_filters == 'from_bank':
            if 0 < bank_ratio < 1:
                bank_size = int(m*bank_ratio)
                filter_bank = Xtr[:bank_size]
                Xtr, Ytr = Xtr[bank_size:], Ytr[bank_size:]
                logging.info(f'Bank size: {bank_size}')
            else:
                raise ValueError(f'Invalid bank_size {bank_size}.')
            filename += f'_br={bank_ratio}'
        elif init_filters == 'from_data':
            filter_bank = Xtr

        if fn:
            filename += f'_{fn}'

        f_proc = []
        if 'c' in fn:
            f_proc.append(center_weight)
        if 'n' in fn:
            f_proc.append(normalize_weight)
        if 'r' in fn:
            f_proc.append(reduce_weight)

        w_gen = WeightFromBankGenerator(filter_bank=filter_bank,
                                        filters_shape=(fs, fs),
                                        filters_shape_high=(fsh, fsh) if fsh else None,
                                        filter_processing=f_proc,
                                        margin=margin,
                                        )
        if wl.startswith('rcc'):
            filters = Filters(n_filters=n_filters,
                              weights_generator=w_gen,
                              activation=activation,
                              maxpool_shape=(maxpool, maxpool))
        elif wl.startswith('rlc'):
            filters = LocalFilters(n_filters=n_filters,
                                   weights_generator=w_gen,
                                   locality=locality,
                                   maxpool_shape=(maxpool, maxpool))

        if wl.endswith('sparseridge'):
            weak_learner = SparseRidgeRC(filters=filters, top_k_filters=top_k)
        elif wl.endswith('ridge'):
            weak_learner = RandomConvolution(filters=filters, weak_learner=Ridge)
        if wl.endswith('ds'):
            weak_learner = RandomConvolution(filters=filters, weak_learner=MulticlassDecisionStump)
            kwargs['n_jobs'] = n_jobs

    else:
        raise ValueError(f'Invalid weak learner name: "{wl}".')

    logging.info(f'Weak learner: {type(weak_learner).__name__}')

    # n_filters_per_layer = [int(f) for f in n_filters_per_layer]
    if len(n_filters_per_layer) == 1:
        n_filters_per_layer = n_filters_per_layer*n_layers

    filename += f'-nfperlayer={n_filters_per_layer}'

    ### Callbacks
    ckpt = ModelCheckpoint(filename=filename+'-{round}.ckpt', dirname='./results', save_last=True)
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
        qb = TransBoost(weak_learner, encoder=encoder)
        qb.fit(Xtr, Ytr, patience=patience,
               n_filters_per_layer=n_filters_per_layer, n_layers=n_layers,
               X_val=X_val, Y_val=Y_val,
               callbacks=callbacks,
               **kwargs)
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
    main()
