import pytest
import numpy as np
import torch

from transboost.aggregation_mechanism import TransformInvariantFeatureAggregation
from transboost.aggregation_mechanism import random_affine


n_examples, n_channels, ex_height, ex_width = 10, 3, 8, 8
X = torch.from_numpy(np.arange(n_examples*n_channels*ex_height*ex_width).reshape(n_examples,n_channels,ex_height, ex_width).astype(np.float32))

n_filters, n_channels, f_height, f_width = 5, 3, 3, 3
n_transforms = 6
class Filters:
    def __init__(self, n_examples=n_examples, n_channels=n_channels, ex_height=ex_height, ex_width=ex_width, n_filters=n_filters, f_height=f_height, f_width=f_width, n_transforms=n_transforms):
        self.weights = np.arange(n_filters*n_channels*f_height*f_width).reshape(n_filters,n_channels,f_height, f_width).astype(np.float32)
        self.pos = [(4, 4)]*n_filters
        self.affine_transforms = [[[random_affine(rotation=15, scale_x=.1, shear_x=10, scale_y=.1, shear_y=10, center=(4,4), angle_unit='degrees') for _ in range(n_channels)] for _ in range(n_transforms)] for _ in range(n_filters)]

class TestTransformInvariantFeatureAggregation:
    def test_call(self):
        filters = Filters()

        tifa = TransformInvariantFeatureAggregation(locality=1)
        high_level_features = tifa(X, filters)
        assert high_level_features.shape == (10, 5)

    def test_compute_padding(self):
        tifa = TransformInvariantFeatureAggregation(locality=1)
        assert tifa._compute_padding(np.ones((5,3,4,4))) == 1
        assert tifa._compute_padding(np.ones((5,3,5,5))) == 2

    def test_get_region_of_interest(self):
        tifa = TransformInvariantFeatureAggregation(locality=1)

        filters = Filters()
        ROI = tifa._get_region_of_interest(X, filters.weights[0], *filters.pos[0])
        assert ROI.shape == (n_examples, n_channels, 5, 5)

        filters = Filters(f_height=5, f_width=5)
        ROI = tifa._get_region_of_interest(X, filters.weights[0], *filters.pos[0])
        assert ROI.shape == (n_examples, n_channels, 5, 5)

        filters = Filters(f_height=6, f_width=6)
        with pytest.raises(ValueError):
            tifa._get_region_of_interest(X, filters.weights[0], *filters.pos[0])

    def test_transform_weights(self):
        filters = Filters()
        tifa = TransformInvariantFeatureAggregation(locality=1)

        pad = int(np.ceil(f_width * (np.sqrt(2)-1)/2)) # = 1 if f_width = 3 or 4
        tw = tifa._transform_weights(filters.weights[0], filters.affine_transforms[0], pad)
        assert tw.shape == (n_transforms, n_channels, f_height+2*pad, f_width+2*pad)
