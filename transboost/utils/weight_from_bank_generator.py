from transboost.aggregation_mechanism.affine_transform import RandomAffineSampler
from transboost.weak_learner import RandomConvolution
import numpy as np
import torch


class Filters:
    def __init__(self, weights, pos, affine_transforms=[]):
        self.pos = pos
        self.weights = weights
        self.affine_transforms = affine_transforms

    def __getitem__(self, item):
        """
        Does not copy the affine transforms since they have to be created at the right scale
        """
        weights = self.weights[item]
        pos = self.pos[item]
        return Filters(weights, pos)



class WeightFromBankGenerator:
    """
    Infinite generator of weights.
    """
    def __init__(self, filter_bank, filters_shape=(5, 5), margin=0,
                 rotation=0, scale=0, shear=0, n_transforms=1):
        """
        Args:
            filter_bank (tensor or array of shape (n_examples, n_channels, height, width)): Bank of images for filters
            to be drawn.

            filters_shape (sequence of 2 integers, optional): Shape of the filters.

            margin (int, optional): Number of pixels from the sides that are excluded from the pool of possible filters.

            rotation (float or tuple of floats): If a single number, a random angle will be picked between (-rotation,
            rotation) uniformly. Else, a random angle will be picked between the two specified numbers.

            scale (float or tuple of floats): Relative scaling factor. If a single number, a random scale factor will be
            picked between (1-scale, 1/(1-scale)) uniformly. Else, a random scale factor will be picked between the two
            specified numbers. Scale factors of x and y axes will be drawn independently from the same range of numbers.

            shear (float or tuple of floats): Shear angle in a direction. If a single number, a random angle will be
            picked between (-shear, shear) uniformly. Else, a random angle will be picked between the two specified
            numbers. Shear degrees of x and y axes will be drawn independently from the same range of numbers.

            n_transforms (int, optional): Number of filters made from the same image (at the same position) but with a
            different random transformation applied each time.
        """
        self.filters_shape = filters_shape
        self.filter_bank = filter_bank
        self.margin = margin
        self.random_affine_sampler = RandomAffineSampler(rotation=rotation, scale_x=scale, scale_y=scale, shear_x=shear,
                                                         shear_y=shear, angle_unit='degrees')
        self.n_transforms = n_transforms

    @property
    def filter_bank(self):
        return self._filter_bank

    @filter_bank.setter
    def filter_bank(self, filter_bank):
        self._filter_bank = RandomConvolution.format_data(filter_bank)
        self.n_examples, n_channels, self.bank_height, self.bank_width = self._filter_bank.shape

    def _draw_filter_shape(self):
        if not self.filters_shape_high:
            return self.filters_shape
        else:
            return (np.random.randint(self.filters_shape[0], self.filters_shape_high[0]),
                    np.random.randint(self.filters_shape[1], self.filters_shape_high[1]))

    def __iter__(self):
        while True:
            height, width = self._draw_filter_shape()
            i_max = self.bank_height - height
            j_max = self.bank_width - width
            yield self._draw_weights_from_bank(height, width, i_max, j_max)

    def draw_n_filters_from_bank(self, n_filters):
        ne, nc, w, h = self.filter_bank.shape
        weights = list()
        pos = list()
        for i in range(n_filters):
            w = list()
            p = list()
            for j in range(nc):
                height, width = self.filter_shape
                i_max = self.bank_height - height
                j_max = self.bank_width - width
                weight, pos = self._draw_weights_from_bank(height, width, i_max, j_max)
                w.append(weight)
                p.append(pos)
            weights.append(w)
            pos.append(p)
        filters = Filters(torch.Tensor(weights), pos)
        return filters

    def _draw_weights_from_bank(self, height, width, i_max, j_max):
        # (i, j) is the top left corner where the filter position was taken
        i, j = (np.random.randint(self.margin, i_max-self.margin),
                np.random.randint(self.margin, j_max-self.margin))
        x = self.filter_bank[np.random.randint(self.n_examples)].clone().detach().cpu()
        weight = x[i:i+width, j:j+height]
        return weight, (i, j)

    def generate_affine_transforms(self, filters):
        affine_transforms = list()
        for nf in filters.pos:
            a = list()
            for nc in nf:
                b = list()
                top_left_c = nc
                center = (top_left_c[0]+(filters.shape[2]-1)/2, top_left_c[1]+(filters.shape[3]-1)/2)
                for k in range(self.n_transforms):
                    affine_transform = self.random_affine_sampler.sample_transformation(center=center)
                    b.append(affine_transform)
                a.append(b)
            affine_transforms.append(a)
        filters.affine_transforms = affine_transforms
