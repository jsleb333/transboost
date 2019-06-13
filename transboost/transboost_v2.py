import torch
from transboost.weak_learner import *
from transboost.label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder

from transboost.callbacks import CallbacksManagerIterator, Step,\
    ModelCheckpoint, CSVLogger, Progression, BestRoundTrackerCallback,\
    BreakOnMaxStepCallback, BreakOnPerfectTrainAccuracyCallback,\
    BreakOnPlateauCallback, BreakOnZeroRiskCallback
from transboost.utils import *
from transboost.quadboost import BoostingRound, QuadBoostMHCR, QuadBoostMH
from torch.nn import functional as F


class Filters:
    def __init__(self, weights, affine_transforms = []):
        self.weights = weights
        self.affine_transforms = affine_transforms


class TransBoost:
    def __init__(self, filter_bank, weak_learner, encoder=None, n_filters_per_layer=100, n_layers=3,
                 f0=None, patience=None, break_on_perfect_train_acc=False, callbacks=None):
        self.filter_bank = filter_bank
        self.weak_learner = weak_learner
        self.encoder = encoder
        self.callbacks = []
        self.weak_predictors = []
        self.filters = []
        self.best_round = None

        self.n_layers = n_layers
        if isinstance(n_filters_per_layer, int):
            self.n_filters_per_layer = [n_filters_per_layer]*n_layers
        else:
            self.n_filters_per_layer = n_filters_per_layer

        if f0 is None:
            self.f0 = np.zeros(self.encoder.encoding_dim)
        else:
            self.f0 = f0

        # Callbacks
        if callbacks is None:
            callbacks = [Progression()]
        elif not any(isinstance(callback, Progression) for callback in callbacks):
            callbacks.append(Progression())

        if break_on_perfect_train_acc:
            callbacks.append(BreakOnPerfectTrainAccuracyCallback())
        if patience:
            callbacks.append(BreakOnPlateauCallback(patience=patience))

        self.callbacks = callbacks

    def algorithm(self, *args, **kwargs):
        return TransBoostAlgorithm(*args, **kwargs)

    def fit(self, X, Y, X_val=None, Y_val=None, **weak_learner_fit_kwargs):
        # Initialization
        self.weak_predictors = []
        if not any(isinstance(callback, BestRoundTrackerCallback) for callback in self.callbacks):
            if X_val is not None and Y_val is not None:
                self.callbacks.append(BestRoundTrackerCallback(quantity='valid_acc'))
            else:
                self.callbacks.append(BestRoundTrackerCallback(quantity='train_acc'))

        # Encodes the labels
        if self.encoder is None:
            self.encoder = OneHotEncoder(Y)
        encoded_Y, weights = self.encoder.encode_labels(Y)

        residue = encoded_Y - self.f0

        self._fit(X, Y, residue, weights, X_val, Y_val, **weak_learner_fit_kwargs)

        return self

    def _fit(self, X, Y, residue, weights, X_val, Y_val, **weak_learner_fit_kwargs):
        encoded_Y_pred = self.predict_encoded(X)
        encoded_Y_val_pred = self.predict_encoded(X_val) if X_val is not None else None

        starting_round = BoostingRound(len(self.weak_predictors))
        boost_manager = CallbacksManagerIterator(self, self.callbacks, starting_round)

        algo = self.algorithm(boost_manager, self.encoder, self.weak_learner,
                                 X, Y, residue, weights, encoded_Y_pred,
                                 X_val, Y_val, encoded_Y_val_pred)
        algo.fit(self.weak_predictors, self.filters, **weak_learner_fit_kwargs)

    def predict(self, X, mode='best'):
        return self.encoder.decode_labels(self.predict_encoded(X, mode))

    def predict_encoded(self, X, mode='last'):
        encoded_Y_pred = np.zeros((X.shape[0], self.encoder.encoding_dim)) + self.f0

        if mode == 'best':
            best = self.best_round.step_number + 1
            wps = self.weak_predictors[:best]
            filters = self.filters[:best]
        else:
            wps = self.weak_predictors
            filters = self.filters
        for wp, f in zip(wps, filters):
            S = get_multi_layers_random_features(X, f)
            encoded_Y_pred += wp.predict(S)
        return encoded_Y_pred

    def evaluate(self, X, Y, return_risk=False, mode='best'):
        encoded_Y_pred = self.predict_encoded(X, mode)
        Y_pred = self.encoder.decode_labels(encoded_Y_pred)

        accuracy = accuracy_score(y_true=Y, y_pred=Y_pred)

        if return_risk:
            encoded_Y, W = self.encoder.encode_labels(Y)
            risk = np.sum(W * (encoded_Y - self.f0 - encoded_Y_pred)**2)

        return accuracy if not return_risk else (accuracy, risk)


class TransBoostAlgorithm:
    n_filters_per_layer: list()

    def __init__(self, boost_manager, encoder, weak_learner, filter_bank,
                 X, Y, residue, weights, encoded_Y_pred,
                 X_val, Y_val, encoded_Y_val_pred,
                 n_filters_per_layer, n_layers=3):
        self.boost_manager = boost_manager
        self.encoder = encoder
        self.weak_learner = weak_learner

        self.X, self.Y, self.residue, self.weights = X, Y, residue, weights
        self.X_val, self.Y_val = X_val, Y_val
        self.encoded_Y_pred = encoded_Y_pred
        self.encoded_Y_val_pred = encoded_Y_val_pred
        self.filter_bank = filter_bank
        self.n_filters_per_layer = n_filters_per_layer
        self.n_layers = n_layers

    def fit(self, weak_predictors, filters, **weak_learner_fit_kwargs):
        with self.boost_manager:  # boost_manager handles callbacks and terminating conditions
            for boosting_round in self.boost_manager:
                this_round_filters = get_multi_layers_filters(self.filter_bank, self.n_filters_per_layer)
                S = get_multi_layers_random_features(self.X, this_round_filters)
                weak_predictor = self.weak_learner().fit(S, self.residue, self.weights, **weak_learner_fit_kwargs)
                weak_prediction = weak_predictor.predict(S)
                self.residue -= weak_prediction
                weak_predictors.append(weak_predictor)
                filters.append(this_round_filters)
                self._evaluate_round(boosting_round, weak_prediction, weak_predictor)

    def _evaluate_round(self, boosting_round, weak_prediction, weak_predictor):
        self.encoded_Y_pred += weak_prediction
        Y_pred = self.encoder.decode_labels(self.encoded_Y_pred)
        boosting_round.train_acc = accuracy_score(y_true=self.Y, y_pred=Y_pred)
        boosting_round.risk = np.sum(self.weights * self.residue**2)

        if not (self.X_val is None or self.Y_val is None or self.encoded_Y_val_pred is None):
            self.encoded_Y_val_pred += weak_predictor.predict(self.X_val)
            Y_val_pred = self.encoder.decode_labels(self.encoded_Y_val_pred)
            boosting_round.valid_acc = accuracy_score(y_true=self.Y_val, y_pred=Y_val_pred)


def advance_to_the_next_layer(X, filters):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    nf, ch, width, height = filters.weights.shape
    output = F.conv2d(X, filters.weights)
    # output.shape -> (n_examples, n_filters, conv_height, conv_width)
    # output = F.max_pool2d(output, (2,2), ceil_mode=True)
    # F.relu(output, inplace=True)
    # output = torch.tanh(output)  # , inplace=True)
    return output


def get_multi_layers_filters(filter_bank, n_filters_per_layer):
    # draw an array of filters Object
    filters = generate_filters_from_bank(filter_bank, sum(n_filters_per_layer))
    multi_layer_filters = []
    multi_layer_filters.append(torch.Tensor(filters[:n_filters_per_layer[0]]))

    filters = filters[n_filters_per_layer[0]:]
    for i, n_filters in enumerate(n_filters_per_layer[1:], 1):
        filters = advance_to_the_next_layer(multi_layer_filters[i - 1], filters)
        multi_layer_filters.append(torch.Tensor(filters[:n_filters]))
        filters = filters[n_filters:]
    return multi_layer_filters


def g_function(X, W):
    pass


def generate_filters_from_bank(filter_bank, n_filters, i_max, j_max):
    i, j = (np.random.randint(i_max, j_max))


def draw_random_affine_transformations(n_transform):
    pass


def get_multi_layers_random_features(examples, filters):
    S = []
    for i in range(len(filters)):
        if i == 0:
            X = examples
        else:
            X = advance_to_the_next_layer(X, filters[i - 1])
        S.append(g_function(X, filters[i]))
    S = torch.cat(S)  # TODO: trouver la bonne fonction pour concatener les tenseurs dans la bonne dim
    return S