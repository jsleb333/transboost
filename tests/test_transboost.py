import pytest
import torch
import numpy
from transboost.transboost_v2 import *
from transboost.utils.weight_from_example_generator import Filters, WeightFromExampleGenerator


@pytest.fixture()
def examples():
    # create Tensor 10x1x20x20
    X = torch.Tensor(numpy.random.randint(0, 255, size=(10, 1, 20, 20)))
    return X


@pytest.fixture()
def filters():
    # create filter object with weights 3x1x5x5 filter
    filter_weights = torch.Tensor(numpy.random.randint(0, 255, size=(3, 1, 5, 5)))
    filters = Filters(filter_weights, [])
    return filters


@pytest.fixture()
def w_gen(examples):
    return WeightFromExampleGenerator(examples)


class Testtransboost:
    def test_get_multi_layers_random_features(self):
        pass

    def test_advance_to_the_next_layer(self,  examples, filters):
        new_X = advance_to_the_next_layer(examples, filters)
        assert new_X.shape == (10, 3, 16, 16)

    def test_get_multi_layers_filters(self, w_gen):
        n_filters_per_layer = [3, 3, 2]
        m_l_f = get_multi_layers_filters(w_gen, n_filters_per_layer)
        assert m_l_f[0].weights.shape == (3, 1, 5, 5)
        assert m_l_f[1].weights.shape == (3, 3, 5, 5)
        assert m_l_f[2].weights.shape == (2, 3, 5, 5)

