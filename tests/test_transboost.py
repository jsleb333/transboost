import pytest
import torch
import numpy
from transboost.transboost_v2 import *


@pytest.fixture()
def examples():
    # create Tensor 10x1x20x20
    X = torch.Tensor(numpy.random.randint(0, 255, size=(10, 1, 20, 20)))
    return X


@pytest.fixture()
def filters():
    # create filter object with weights 3x1x5x5 filter
    filter_weights = torch.Tensor(numpy.random.randint(0, 255, size=(3, 1, 5, 5)))
    filters = Filters(filter_weights)
    return filters

@pytest.fixture()
def transboot_algo():
    TransBoostAlgorithm()


class Testtransboost:
    def test_get_multi_layers_random_features(self):
        pass

    def test_advance_to_the_next_layer(self,  examples, filters):
        new_X = advance_to_the_next_layer(examples, filters)
        assert new_X.shape == (10, 3, 16, 16)

    def test_generate_filters_from_bank(self):
        pass

    def test_init_filters(self):
        pass

    def test_get_multi_layers_filters(self):
        pass

