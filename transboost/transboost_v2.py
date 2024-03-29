import torch
from transboost.weak_learner import *
from transboost.label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder

from transboost.callbacks import CallbacksManagerIterator, Step, ModelCheckpoint, CSVLogger,\
    Progression, BestRoundTrackerCallback,BreakOnMaxStepCallback, \
    BreakOnPerfectTrainAccuracyCallback, BreakOnPlateauCallback, BreakOnZeroRiskCallback
from transboost.utils import FiltersGenerator
from torch.nn import functional as F
from transboost.aggregation_mechanism import TransformInvariantFeatureAggregation as Tifa


class TransBoost:
    def __init__(self, filters_generator, weak_learner, aggregation_mechanism,
                 encoder=None, n_filters_per_layer=100, n_layers=3, f0=None,
                 patience=None, break_on_perfect_train_acc=False, callbacks=None):
        """
        Args:
            filters_generator (FiltersGenerator object): Objects that generates filters from a bank of examples.

            weak_learner (Object that defines the 'fit' method and the 'predict' method): Weak learner that generates weak predictors to be boosted on.

            encoder (LabelEncoder object, optional): Object that encodes the labels to provide an easier separation problem. If None, a one-hot encoding is used.

            f0 (Array of shape (encoding_dim,), optional, default=None): Initial prediction function. If None, f0 is set to 0.

            patience (int, optional, default=None): Number of boosting rounds before terminating the algorithm when the training accuracy shows no improvements. If None, the boosting rounds will continue until max_round_number iterations (if not None).

            break_on_perfect_train_acc (Boolean, optional, default=False): If True, it will stop the iterations if a perfect train accuracy of 1.0 is achieved.

            callbacks (Iterable of Callback objects, optional, default=None): Callbacks objects to be called at some specific step of the training procedure to execute something. Ending conditions of the boosting iteration are handled with BreakCallbacks. If callbacks contains BreakCallbacks and terminating conditions (max_round_number, patience, break_on_perfect_train_acc) are not None, all conditions will be checked at each round and the first that is not verified will stop the iteration.

        """
        self.filters_generator = filters_generator
        self.weak_learner = weak_learner
        self.aggregation_mechanism = aggregation_mechanism
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
        """
        Function that fits the model to the data.

        The function is split into two parts: the first prepare the data, the second, done in _fit, actually executes the algorithm. The iteration and the callbacks are handled by a CallbacksManagerIterator.

        Args:
            X (Array of shape (n_examples, ...)): Examples.

            Y (Iterable of 'n_examples' elements): Labels for the examples X. Y is encoded with the encode_labels method if one is provided, else it is transformed as one-hot vectors.

            X_val (Array of shape (n_val, ...), optional, default=None): Validation examples. If not None, the validation accuracy will be evaluated at each boosting round.

            Y_val (Iterable of 'n_val' elements, optional, default=None): Validation labels for the examples X_val. If not None, the validation accuracy will be evaluated at each boosting round.

            weak_learner_fit_kwargs: Keyword arguments to pass to the fit method of the weak learner.

        Returns self.
        """
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
        """
        functions that fit fits the model to the data. Should not be used as is but wrapped in the fit function
        """
        encoded_Y_pred = self.predict_encoded(X)
        encoded_Y_val_pred = self.predict_encoded(X_val) if X_val is not None else None

        starting_round = BoostingRound(len(self.weak_predictors))
        boost_manager = CallbacksManagerIterator(self, self.callbacks, starting_round)

        algo = self.algorithm(boost_manager, self.encoder, self.weak_learner,
                              self.filters_generator, self.aggregation_mechanism,
                              X, Y, residue, weights, encoded_Y_pred,
                              X_val, Y_val, encoded_Y_val_pred,
                              self.n_filters_per_layer, self.n_layers)
        algo.fit(self.weak_predictors, self.filters, **weak_learner_fit_kwargs)

    def predict(self, X, mode='best'):
        """
        predict the labels of the examples X
        :param X: examples
        :param mode: 'best' will only use the weights used by the best round and 'last' will use all the weights
        :return: (Tensor) decoded labels
        """
        return self.encoder.decode_labels(self.predict_encoded(X, mode))

    def predict_encoded(self, X, mode='last'):
        """
        predict the encoded labels f the examples X
        :param X: (Tensor) examples
        :param mode: 'best' will only use the weights used by the best round and 'last' will use all the weights
        :return: (Iterable) encoded labels
        """
        encoded_Y_pred = np.zeros((X.shape[0], self.encoder.encoding_dim)) + self.f0

        if mode == 'best':
            best = self.best_round.step_number + 1
            wps = self.weak_predictors[:best]
            filters = self.filters[:best]
        else:
            wps = self.weak_predictors
            filters = self.filters
        for wp, f in zip(wps, filters):
            S = self.aggregation_mechanism(X, f)
            encoded_Y_pred += wp.predict(S)
        return encoded_Y_pred

    def evaluate(self, X, Y, return_risk=False, mode='best'):
        """
        evaluate the accuracy of the trained algorithm
        :param X: examples to predict
        :param Y: labels
        :param return_risk: if True, the function will also calculate the quadratic risk of this prediction
        :param mode: 'best' will only use the weights used by the best round and 'last' will use all the weights
        :return: acccuracy (double) or (accuracy, risk)
        """
        encoded_Y_pred = self.predict_encoded(X, mode)
        Y_pred = self.encoder.decode_labels(encoded_Y_pred)

        accuracy = accuracy_score(y_true=Y, y_pred=Y_pred)

        if return_risk:
            encoded_Y, W = self.encoder.encode_labels(Y)
            risk = np.sum(W * (encoded_Y - self.f0 - encoded_Y_pred)**2)

        return accuracy if not return_risk else (accuracy, risk)


class TransBoostAlgorithm:
    """
    This is an implementation of the TransBoost algorithm. It is intended to be used inside the TransBoost class API and not as is.
    """
    def __init__(self, boost_manager, encoder, weak_learner, filters_generator,
                 aggregation_mechanism,
                 X, Y, residue, weights, encoded_Y_pred,
                 X_val, Y_val, encoded_Y_val_pred,
                 n_filters_per_layer=list(), n_layers=3):
        self.boost_manager = boost_manager
        self.encoder = encoder
        self.weak_learner = weak_learner
        self.aggregation_mechanism = aggregation_mechanism

        self.X, self.Y, self.residue, self.weights = X, Y, residue, weights
        self.X_val, self.Y_val = X_val, Y_val
        self.encoded_Y_pred = encoded_Y_pred
        self.encoded_Y_val_pred = encoded_Y_val_pred
        self.filters_generator = filters_generator
        self.n_filters_per_layer = n_filters_per_layer
        self.n_layers = n_layers

    def fit(self, weak_predictors, filters, **weak_learner_fit_kwargs):
        """
        Executes the algorithm.
        Appends the weak_predictors list with the fitted weak learners.

        Args:
            weak_predictors (list): Reference to the list of weak_predictors of the model.
            filters (list of Filters object): Filters used in the aggregation mechanism.
            weak_learner_fit_kwargs: Keyword arguments needed to fit the weak learner.

        Returns None.
        """
        with self.boost_manager:  # boost_manager handles callbacks and terminating conditions
            for boosting_round in self.boost_manager:
                this_round_filters = get_multi_layers_filters(self.filters_generator, self.n_filters_per_layer)
                S = self.aggregation_mechanism(self.X, this_round_filters)
                weak_predictor = self.weak_learner().fit(S, self.residue, self.weights, **weak_learner_fit_kwargs)
                weak_prediction = weak_predictor.predict(S)
                self.residue -= weak_prediction
                weak_predictors.append(weak_predictor)
                filters.append(this_round_filters)
                self._evaluate_round(boosting_round, weak_prediction, weak_predictor, this_round_filters)

    def _evaluate_round(self, boosting_round, weak_prediction, weak_predictor, filters):
        """
        Compute the metrics for the current boosting round
        Appends the weak_predictors and weak_predictors_weights lists with the fitted weak learners.

        Args:
            boosting round (BoostingRound Object): Object use to store information on the metrics
            weak_prediction (List): Reference to the predictions
            weak_predictor(WeakLearner Object): Reference to the trained weak learner for the current round
            filters(Filters Object) : filters used to get the multlayers attribute for this round

        Returns None.
        """
        self.encoded_Y_pred += weak_prediction
        Y_pred = self.encoder.decode_labels(self.encoded_Y_pred)
        boosting_round.train_acc = accuracy_score(y_true=self.Y, y_pred=Y_pred)
        boosting_round.risk = np.sum(self.weights * self.residue**2)

        if not (self.X_val is None or self.Y_val is None or self.encoded_Y_val_pred is None):
            S = self.aggregation_mechanism(self.X_val, filters)
            self.encoded_Y_val_pred += weak_predictor.predict(S)
            Y_val_pred = self.encoder.decode_labels(self.encoded_Y_val_pred)
            boosting_round.valid_acc = accuracy_score(y_true=self.Y_val, y_pred=Y_val_pred)


def advance_to_the_next_layer(X, filters):
    """
    Transform the examples by using convolution with filters. The
    Args:
        X (Array of shape (n_examples, ...)): Examples to advance to the next layers
        filters (Filters object): Filters to be convoluted with the example

    Returns the transformed examples (Tensor of shape (n_examples, n_filters, ...)).
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    weights = filters.weights.to(device=X.device)
    next_layer = F.conv2d(X, weights)
    next_layer -= torch.mean(next_layer, dim=0)
    maxim = torch.unsqueeze(torch.max(next_layer, dim=0)[0], dim=0)
    abs_min =torch.unsqueeze(torch.abs(torch.min(next_layer, dim=0)[0]), dim=0)
    scale = torch.max(torch.cat((maxim, abs_min), dim=0), dim = 0)[0]
    scale = torch.where(scale <= 1, torch.ones_like(scale), scale)

    next_layer /= scale
    # n_filters, n_channels, width, height = filters.weights.shape
    # next_layer.shape -> (n_examples, n_filters, conv_height, conv_width)
    # next_layer = F.max_pool2d(next_layer, (2,2), ceil_mode=True)
    # F.relu(next_layer, inplace=True)
    # next_layer = torch.tanh(next_layer)  # , inplace=True)
    return next_layer


def get_multi_layers_filters(filters_generator: FiltersGenerator, n_filters_per_layer):
    """
    generate filters for each layers by convolving filters together:
    filters of the second layers are obtaine by convolving filters of the first layer together
    filters of the third layers are obtained by convolving filter of the second layer together

    Args:
        filters_generator (FiltersGenerator Object): Object that generate filters by drawing them in a bank of exameples and apply
        n_filters_per_layer (Array)

    Returns the filters  (Array of array of filter objects (one array per layer)).
    """
    # filters_generator first contains the the examples from the original examples distributon
    # draw numbers representing the examples that will generate filters
    examples = filters_generator.draw_n_examples_from_bank(sum(n_filters_per_layer))
    first_layer_filters = filters_generator.generate_filters(examples[:n_filters_per_layer[0]])
    multi_layer_filters = [first_layer_filters]
    examples = examples[n_filters_per_layer[0]:]

    for i, n_filters in enumerate(n_filters_per_layer[1:]):
        examples = advance_to_the_next_layer(examples, multi_layer_filters[i])
        next_layer_filters = filters_generator.generate_filters(examples[:n_filters])
        multi_layer_filters.append(next_layer_filters)
        examples = examples[n_filters:]

    return multi_layer_filters


class MultiLayersRandomFeatures:
    """
    Class that can be initialized with hyperparameters that will be used if the feature aggregation mechanisms
    This class can then be called to create the random features with the feature aggregation mechanism
    """
    def __init__(self, locality=3, maxpool_shape=(3,3), activation=None):
        self.locality = locality
        self.maxpool_shape = maxpool_shape
        self.activation = activation

    def __call__(self, examples, filters):
        """
        generate random features by using the tifa to aggregate examples with filters
        Args:
            examples (Array of shape (n_examples, ...)): Examples to aggregate
            filters (Array of array of filters object (one array per layer)): obtained with the method get_multi_layer_filters

        Returns the multilayer random features  (Array of shape (sum(n_fiters_per_layers))).
        """
        S = []
        tifa = Tifa(locality=self.locality,
                    maxpool_shape=self.maxpool_shape,
                    activation=self.activation)
        for i in range(len(filters)):
            if i == 0:
                X = examples
            else:
                X = advance_to_the_next_layer(X, filters[i - 1])
            S.append(tifa(X, filters[i]))
        S = torch.cat(S, dim=1)
        return S


class BoostingRound(Step):
    """
    Class that stores information about the current boosting round like the the round number and the training and validation accuracies. Used by the CallbacksManagerIterator in the _QuadBoostAlgorithm.fit method.
    """
    def __init__(self, round_number=0):
        super().__init__(step_number=round_number)
        self.train_acc = None
        self.valid_acc = None
        self.risk = None
