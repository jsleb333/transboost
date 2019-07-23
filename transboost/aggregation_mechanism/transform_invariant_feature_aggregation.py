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
            self.maxpool_shape = [-1, *maxpool_shape] if len(maxpool_shape) == 2 else list(maxpool_shape)

    def __call__(self, X, filters):
        """
        Extracts transform invariant features of the examples X from the filters W.

        Args:
            X (Array or Tensor of shape (n_examples, n_channels, height, width)): Examples to extract high level features from.
            filters (Filter object): Filters used to extract high level features.
        """
        n_examples, n_channels, height, width = X.shape
        high_level_features = []
        # filters.weight.shape = (n_filters, n_channels, filter_height, filter_width)
        for weights, pos, ats in zip(filters.weights, filters.pos, filters.affine_transforms):
            pad = self._compute_padding(weights)

            transformed_weights = self._transform_weights(weights, ats, pad)
            # transformed_weights.shape: (n_transforms, n_ch, filter_height+pad, filter_width+pad)
            self.n_transforms, *_ = transformed_weights.shape
            transformed_weights = transformed_weights.to(device=X.device)

            ROI = self._get_region_of_interest(X, weights, pad, *pos)

            output = F.conv2d(ROI, transformed_weights)
            # output.shape = (n_examples, n_transforms, height, width)
            if self.maxpool_shape:
                output = torch.unsqueeze(output, dim=1)
                # output.shape = (n_examples, 1, n_transforms, height, width)
                maxpool_shape = self._compute_maxpool_shape(output)
                output = F.max_pool3d(output, maxpool_shape, ceil_mode=True)

            high_level_features.append(output.reshape(n_examples, -1))

        high_level_features = torch.cat(high_level_features, dim=1)
        if self.activation:
            high_level_features = self.activation(high_level_features)
        high_level_features = high_level_features.cpu().reshape((n_examples, -1))
        return high_level_features

    def _compute_padding(self, weights):
        """
        :param weights: (Array or Tensor of shape (n_examples, n_channels, height, width)): Weights to use for exreaction
        :return: (int): the padding to use for trasforming the weight
        """
        filter_width = weights.shape[-1]
        PAD_FACTOR = (np.sqrt(2)-1)/2# Minimum factor to pad to not loose any pixels when rotating by π/4 a square filter.
        pad = int(np.ceil(filter_width * PAD_FACTOR))
        return pad

    def _get_region_of_interest(self, X, weights, pad, i, j):
        """
        return the region of interest on the examples on which to extract information
        :param X: X (Array or Tensor of shape (n_examples, n_channels, height, width)): Examples to extract high level features from.
        :param weights: Array or Tensor of shape ( n_channels, height, width)): Weights to use for extraction
        :param pad: padding
        :param i: x position of the filter
        :param j: y position of the filter
        :return: X (Array or Tensor of shape (n_examples, n_channels, height, width)): Examples cropped around the region of interest
        """
        n_examples, n_channels, height, width = X.shape
        i_min = max(i - self.locality - pad, 0)
        j_min = max(j - self.locality - pad, 0)
        i_max = min(i + weights.shape[-2] + self.locality + pad, height)
        j_max = min(j + weights.shape[-1] + self.locality + pad, width)

        if j_max - j_min < weights.shape[-1] or i_max - i_min < weights.shape[-2]:
            raise ValueError(f"Filter shape of {(weights.shape[-2], weights.shape[-1])} too large.")

        return X[:,:,i_min:i_max, j_min:j_max]

    def _transform_weights(self, weights, affine_transforms, pad):
        """
        Apply the affine transform to the weights
        :param weights: Array or Tensor of shape ( n_channels, height, width)): Weights to use for extraction
        :param affine_transforms: Iterable of AffineTransform object of shape (n_transform, n_channel)
        :param pad: padding
        :return: transformed weights Array or Tensor of shape ( n_transform, n_channels, height, width)): Weights to use for extraction
        """
        transformed_weights = []
        for affine_transforms_ch in affine_transforms:
            transformed_chs = []
            for ch, affine_transform in zip(weights, affine_transforms_ch):
                affine_transform.center += pad
                transformed_ch = affine_transform(np.pad(ch, pad, 'constant')) / affine_transform.determinant
                transformed_ch = torch.unsqueeze(torch.from_numpy(transformed_ch), dim=0)
                transformed_chs.append(transformed_ch)
            transformed_chs = torch.unsqueeze(torch.cat(transformed_chs, dim=0), dim=0)
            transformed_weights.append(transformed_chs)

        return torch.cat(transformed_weights, dim=0)

    def _compute_maxpool_shape(self, output):
        """
        :param output: Array or Tensor of shape (n_transform n_channels, height, width)): result of the convolution
        :return: maxpool_shape
        """
        maxpool_shape = [i for i in self.maxpool_shape]

        if maxpool_shape[0] == -1:
            maxpool_shape[0] = self.n_transforms
        if maxpool_shape[1] == -1:
            maxpool_shape[1] = output.shape[-2]
        if maxpool_shape[2] == -1:
            maxpool_shape[2] = output.shape[-1]

        return maxpool_shape
