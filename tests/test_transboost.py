import pytest
import torch
import numpy
from transboost.transboost_v2 import *
from transboost.utils import Filters, FiltersGenerator


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

n_transforms = 9

@pytest.fixture()
def filter_generator(examples):
    return FiltersGenerator(examples, n_transforms=n_transforms)


class Testtransboost:
    def test_get_multi_layers_random_features(self):
        pass

    def test_advance_to_the_next_layer(self,  examples, filters):
        new_X = advance_to_the_next_layer(examples, filters)
        assert new_X.shape == (10, 3, 16, 16)

    def test_get_multi_layers_filters(self, filter_generator):
        n_filters_per_layer = [3, 3, 2]
        mlf = get_multi_layers_filters(filter_generator, n_filters_per_layer)
        assert mlf[0].weights.shape == (3, 1, 5, 5)
        assert len(mlf[0].pos) == 3
        assert len(mlf[0].affine_transforms) == 3
        assert len(mlf[0].affine_transforms[0]) == n_transforms
        assert len(mlf[0].affine_transforms[0][0]) == 1

        assert mlf[1].weights.shape == (3, 3, 5, 5)
        assert len(mlf[1].pos) == 3
        assert len(mlf[1].affine_transforms) == 3
        assert len(mlf[1].affine_transforms[0]) == n_transforms
        assert len(mlf[1].affine_transforms[0][0]) == 3

        assert mlf[2].weights.shape == (2, 3, 5, 5)
        assert len(mlf[2].pos) == 2
        assert len(mlf[2].affine_transforms) == 2
        assert len(mlf[2].affine_transforms[0]) == n_transforms
        assert len(mlf[2].affine_transforms[0][0]) == 3
