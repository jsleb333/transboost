import pytest
import numpy as np
import torch

from transboost.aggregation_mechanism import TransformInvariantFeatureAggregation
from transboost.aggregation_mechanism import random_affine


class TestTransformInvariantFeatureAggregation:
    def test_call(self):
        n_examples, n_channels, ex_height, ex_width = 10, 3, 8, 8
        X = torch.from_numpy(np.arange(n_examples*n_channels*ex_height*ex_width).reshape(n_examples,n_channels,ex_height, ex_width).astype(np.float32))

        n_filters, n_channels, f_height, f_width = 5, 3, 4, 4
        n_transforms = 6
        class Filters:
            weights = np.arange(n_filters*n_channels*f_height*f_width).reshape(n_filters,n_channels,f_height, f_width).astype(np.float32)
            pos = [(4, 4)]*n_filters
            affine_transforms = [[[random_affine(rotation=15, scale_x=.1, shear_x=10, scale_y=.1, shear_y=10, center=(4,4), angle_unit='degrees') for i in range(n_channels)] for _ in range(n_filters)] for x in range(n_transforms)]
        filters = Filters()
        print(len(filters.affine_transforms))
        print(len(filters.affine_transforms[0]))

        tifa = TransformInvariantFeatureAggregation()
        tifa(X, filters)
