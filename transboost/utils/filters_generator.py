from transboost.aggregation_mechanism.affine_transform import RandomAffineSampler, AffineTransform
import numpy as np
import torch


class Filters:
    def __init__(self, weights, pos, affine_transforms=list()):
        self.pos = pos
        self.weights = weights
        self.affine_transforms = affine_transforms

    def __getitem__(self, item):
        """
        Does not copy the affine transforms since they have to be created at the right scale
        """
        weights = self.weights[item]
        pos = self.pos[item]
        affine_transform = self.affine_transforms[item]
        return Filters(weights, pos, affine_transform)


class FiltersGenerator:
    """
    Infinite generator of weights.
    """
    def __init__(self, filter_bank, filters_shape=(5, 5), margin=0,
                 rotation=0, scale=0, shear=0, n_transforms=1, filters_preprocessing=None):
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

            filters_preprocessing (list of callables or None): Callables to be applied on the weights of the filters. Each callable will receive weights and should return preprocessed weights in order.
        """
        self.filters_shape = filters_shape
        self.filter_bank = filter_bank
        self.margin = margin
        self.random_affine_sampler = RandomAffineSampler(rotation=rotation,
                                                         scale_x=scale, scale_y=scale,
                                                         shear_x=shear, shear_y=shear,
                                                         angle_unit='degrees')
        self.n_transforms = n_transforms
        self.filters_preprocessing = filters_preprocessing or []

    @property
    def filter_bank(self):
        return self._filter_bank

    @filter_bank.setter
    def filter_bank(self, filter_bank):
        self._filter_bank = self.format_data(filter_bank)
        self.n_examples, n_channels, self.bank_height, self.bank_width = self._filter_bank.shape

    def draw_n_examples_from_bank(self, n_examples):
        examples = list()
        for i in range(n_examples):
            example = self.filter_bank[np.random.randint(self.n_examples)].clone().detach().cpu()
            example = torch.unsqueeze(example, dim=0)
            examples.append(example)
        return torch.cat(examples, dim=0)

    def generate_filters(self, examples):
        weights, pos = [], []
        for example in examples:
            weight, p = self._generate_filter(example)
            for preprocessing in self.filters_preprocessing:
                weight = preprocessing(weight)
            weights.append(torch.unsqueeze(weight, dim=0))
            pos.append(p)
        weights = torch.cat(weights, dim=0)
        affine_transforms = self._generate_affine_transforms(weights, pos, self.n_transforms)
        filters = Filters(weights, pos, affine_transforms)
        return filters

    def _generate_filter(self, example):
        # (i, j) is the top left corner where the filter position was taken
        height, width = self.filters_shape
        i_max = example.shape[1] - height
        j_max = example.shape[2] - width
        i, j = (np.random.randint(self.margin, i_max-self.margin),
                np.random.randint(self.margin, j_max-self.margin))
        x = example.narrow(1, i, width)
        x = x.narrow(2, j, height)
        return x, (i, j)

    def _generate_affine_transforms(self, filters_weights, filters_pos, n_transforms):
        n_filters, n_channels, height, width = filters_weights.shape
        center = ((height - 1) / 2, (width - 1) / 2)
        if not n_transforms:
            affine_transforms = [
                [
                    [
                        AffineTransform()
                        for _ in range(n_channels)]
                    ]
                for pos in filters_pos]
        else:
            affine_transforms = [
                [
                    [
                        self.random_affine_sampler.sample_transformation(center)
                        for _ in range(n_channels)]
                    for _ in range(n_transforms)]
                for pos in filters_pos]

        return affine_transforms

    @staticmethod
    def format_data(data):
        """
        Formats a data array to the right format accepted by this class, which is a torch.Tensor of shape (n_examples, n_channels, height, width).
        """
        if type(data) is np.ndarray:
            data = torch.from_numpy(data).float()
        if len(data.shape) == 3:
            data = torch.unsqueeze(data, dim=1)
        return data


def center_weight(weight):
    mean = torch.mean(weight)
    weight -= mean
    return weight


def normalize_weight(weight):
    weight /= torch.norm(weight, p=2)
    return weight


def reduce_weight(weight):
    weight /= torch.std(weight)
    return weight
