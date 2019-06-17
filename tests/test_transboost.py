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
def w_gen(examples):
    return FiltersGenerator(examples, n_transforms=n_transforms)


class Testtransboost:
    def test_get_multi_layers_random_features(self):
        pass

    def test_advance_to_the_next_layer(self,  examples, filters):
        new_X = advance_to_the_next_layer(examples, filters)
        assert new_X.shape == (10, 3, 16, 16)

    def test_get_multi_layers_filters(self, w_gen):
        n_filters_per_layer = [4, 6, 2]
        mlf = get_multi_layers_filters(w_gen, n_filters_per_layer)
        assert mlf[0].weights.shape == (4, 1, 5, 5)
        assert len(mlf[0].pos) == 4
        assert len(mlf[0].affine_transforms) == 4
        assert len(mlf[0].affine_transforms[0]) == 1
        assert len(mlf[0].affine_transforms[0][0]) == n_transforms

        assert mlf[1].weights.shape == (6, 4, 5, 5)
        assert len(mlf[1].pos) == 6
        assert len(mlf[1].affine_transforms) == 6
        assert len(mlf[1].affine_transforms[0]) == 4
        assert len(mlf[1].affine_transforms[0][0]) == n_transforms

        assert mlf[2].weights.shape == (2, 6, 5, 5)
        assert len(mlf[2].pos) == 2
        assert len(mlf[2].affine_transforms) == 2
        assert len(mlf[2].affine_transforms[0]) == 6
        assert len(mlf[2].affine_transforms[0][0]) == n_transforms
