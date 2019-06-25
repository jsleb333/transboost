import pytest
import torch

from transboost.utils import *


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
