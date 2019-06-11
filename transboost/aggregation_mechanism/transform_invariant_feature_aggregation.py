import numpy as np
from sklearn.linear_model import Ridge
import inspect
import torch
from torch import nn
from torch.nn import functional as F
import warnings
import matplotlib.pyplot as plt
from graal_utils import timed
from .affine_transform import RandomAffine


class TransformInvariantFeatureAggregation:
    def __init__(self, locality=3, maxpool_shape=(3,3), activation=None):
        """
        Args:
            locality (int, optional): Applies the filters locally around the place where the filter was taken from the picture in the filter bank. For example, for a filters_shape=(N,N) and a locality=L, the convolution will be made on a square of side N+2L centered around the original position. It will yield an array of side 2L+1. No padding is made in case the square exceeds the size of the examples.

            maxpool_shape (tuple of 2 or 3 integers or None, optional): Shape of the maxpool kernel layer. If None, no maxpool is done. If one of the dim is set to -1, the maxpool is done over all components of this dim. If there are 3 components, the first is used for the transformations.

            activation (Callable or None, optional): Activation function to apply which returns transformed data.
        """

        self.locality = locality
        self.activation = activation
        self.maxpool_shape = maxpool_shape
        if maxpool_shape:
            self.maxpool_shape = [1, *maxpool_shape] if len(maxpool_shape) == 2 else list(maxpool_shape)

    def __call__(self, X, filters):
        """
        Extracts transform invariant features of the examples X from the filters W.

        Args:
            X (Array or Tensor of shape (n_examples, n_channels, height, width)): Examples to extract high level features from.
            filters (Filter object): Filters used to extract high level features.
        """
        filters = [filters.to(device=X.device) for f in filters]

        n_examples, n_channels, height, width = X.shape
        random_features = []
        for conv_filter in filters:
            i_min = max(conv_filter.i - self.locality, 0)
            j_min = max(conv_filter.j - self.locality, 0)
            i_max = min(conv_filter.i + conv_filter.weight.shape[-2] + self.locality, height)
            j_max = min(conv_filter.j + conv_filter.weight.shape[-1] + self.locality, width)

            output = F.conv2d(X[:,:,i_min:i_max, j_min:j_max], conv_filter.weight)
            # output.shape -> (n_examples, n_transforms, height, array)
            if maxpool_shape:
                output = torch.unsqueeze(output, dim=1)
                # output.shape -> (n_examples, 1, n_transforms, height, array)
                self._compute_maxpool_shape(output)
                output = F.max_pool3d(output, maxpool_shape, ceil_mode=True)

            random_features.append(output.reshape(n_examples, -1))

        random_features = torch.cat(random_features, dim=1)
        random_features = self.activation(random_features)
        random_features = self.activation(random_features)
        random_features = random_features.cpu().reshape((n_examples, -1))
        return random_features

    def _compute_maxpool_shape(self, output):
        if self.maxpool_shape[0] == -1:
            self.maxpool_shape[0] = self.n_transforms
        if self.maxpool_shape[1] == -1:
            self.maxpool_shape[1] = output.shape[-2]
        if self.maxpool_shape[2] == -1:
            self.maxpool_shape[2] = output.shape[-1]
