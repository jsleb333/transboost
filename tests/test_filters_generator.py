import pytest
import torch

from transboost.utils import *
from transboost.aggregation_mechanism import AffineTransform


@pytest.fixture
def bank():
    return torch.randn(10,3,5,5)

class TestFiltersGenerator:
    def test_weights_have_zero_mean(self, bank):
        fp = [center_weight]
        fg = FiltersGenerator(filter_bank=bank, filters_shape=(3,3), filters_preprocessing=fp)
        chosen_x = fg.draw_n_examples_from_bank(3)
        filters = fg.generate_filters(chosen_x)
        assert torch.abs(torch.mean(filters.weights)) <= 10e-6 # tolerance

    def test_generate_five_affine_transforms(self, bank):
        n_transforms = 5
        fg = FiltersGenerator(filter_bank=bank, filters_shape=(3,3))
        chosen_x = fg.draw_n_examples_from_bank(4)
        weights = chosen_x[:,:,1:4,1:4]
        pos = [(2,2)]*4
        afs = fg._generate_affine_transforms(weights, pos, n_transforms)
        assert len(afs) == 4
        assert len(afs[0]) == n_transforms
        assert len(afs[0][0]) == 3

    def test_generate_zero_affine_transforms(self, bank):
        n_transforms = 0
        fg = FiltersGenerator(filter_bank=bank, filters_shape=(3,3))
        chosen_x = fg.draw_n_examples_from_bank(4)
        weights = chosen_x[:,:,1:4,1:4]
        pos = [(2,2)]*4
        afs = fg._generate_affine_transforms(weights, pos, n_transforms)
        assert len(afs) == 4
        assert len(afs[0]) == 1
        assert len(afs[0][0]) == 3
        assert isinstance(afs[0][0][0], AffineTransform)
        assert not afs[0][0][0] # Is identity transformation
