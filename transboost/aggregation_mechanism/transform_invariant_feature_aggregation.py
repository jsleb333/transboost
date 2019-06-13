import numpy as np
from sklearn.linear_model import Ridge
import inspect
import torch
from torch import nn
from torch.nn import functional as F
import warnings
import matplotlib.pyplot as plt
from graal_utils import timed

from transboost.aggregation_mechanism import random_affine


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
        n_examples, n_channels, height, width = X.shape
        high_level_features = []
        # filters.weight.shape = (n_filters, n_channel, filter_height, filter_width)
        for weights, (i,j), ats in zip(filters.weights, filters.pos, filters.affine_transforms):
            width = weights.shape[-1]
            FACTOR = (np.sqrt(2)-1)/2# Minimum factor to pad to not loose any pixels when rotating by π/4 a square filter.
            pad = int(np.ceil(width * FACTOR))
            transformed_weights = self._transform_weights(weights, ats, pad)
            # transformed_weights.shape = (n_transforms, n_channel, filter_height+pad, filter_width+pad)
            transformed_weights.to(device=X.device)

            i_min = max(i - self.locality, 0)
            j_min = max(j - self.locality, 0)
            i_max = min(i + transformed_weights.shape[-2] + self.locality, height)
            j_max = min(j + transformed_weights.shape[-1] + self.locality, width)

            output = F.conv2d(X[:,:,i_min:i_max, j_min:j_max], transformed_weights)
            # output.shape = (n_examples, n_transforms, height, array)
            if self.maxpool_shape:
                output = torch.unsqueeze(output, dim=1)
                # output.shape = (n_examples, 1, n_transforms, height, array)
                self._compute_maxpool_shape(output)
                output = F.max_pool3d(output, self.maxpool_shape, ceil_mode=True)

            high_level_features.append(output.reshape(n_examples, -1))

        high_level_features = torch.cat(high_level_features, dim=1)
        if self.activation:
            high_level_features = self.activation(high_level_features)
        high_level_features = high_level_features.cpu().reshape((n_examples, -1))
        return high_level_features

    def _transform_weights(self, weights, affine_transforms, pad):
        transformed_weights = []
        for affine_transforms_ch in affine_transforms:
            transformed_chs = []
            for ch, affine_transform in zip(weights, affine_transforms_ch):
                transformed_ch = affine_transform(np.pad(ch, pad, 'constant')) / affine_transform.determinant
                transformed_ch = torch.unsqueeze(torch.from_numpy(transformed_ch), dim=0)
                transformed_chs.append(transformed_ch)
            transformed_chs = torch.unsqueeze(torch.cat(transformed_chs, dim=0), dim=0)
            transformed_weights.append(transformed_chs)

        return torch.cat(transformed_weights, dim=0)

    def _compute_maxpool_shape(self, output):
        if self.maxpool_shape[0] == -1:
            self.maxpool_shape[0] = self.n_transforms
        if self.maxpool_shape[1] == -1:
            self.maxpool_shape[1] = output.shape[-2]
        if self.maxpool_shape[2] == -1:
            self.maxpool_shape[2] = output.shape[-1]
