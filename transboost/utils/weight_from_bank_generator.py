class WeightFromBankGenerator:
    """
    Infinite generator of weights.
    """
    def __init__(self, filter_bank, filters_shape=(5, 5), filters_shape_high=None, margin=0, filter_processing=None,
                 rotation=0, scale=0, shear=0, padding=2, n_transforms=1):
        """
        Args:
            filter_bank (tensor or array of shape (n_examples, n_channels, height, width)): Bank of images for filters
            to be drawn.

            filters_shape (sequence of 2 integers, optional): Shape of the filters.

            filters_shape_high (sequence of 2 integers or None, optional): If not None, the shape of the filters will be
            randomly drawn from a uniform distribution between filters_shape (inclusive) and filters_shape_high
            (exclusive).

            margin (int, optional): Number of pixels from the sides that are excluded from the pool of possible filters.

            filter_processing (callable or iterable of callables or None, optional): Callable or iterable of callables
            that execute (sequentially) some process on one weight and returns the result.

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
        self.filters_shape_high = filters_shape_high
        self.filter_bank = filter_bank
        self.margin = margin
        if callable(filter_processing): filter_processing = [filter_processing]
        self.filter_processing = filter_processing or []

        self.rotation, self.scale, self.shear = rotation, scale, shear
        self.random_affine_sampler = RandomAffine(rotation=rotation, scale_x=scale, scale_y=scale, shear_x=shear,
                                                  shear_y=shear, angle_unit='degrees')
        self.padding = padding
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
            yield self._draw_from_bank(height, width, i_max, j_max)

    def _draw_from_bank(self, height, width, i_max, j_max):
        # (i, j) is the top left corner where the filter position was taken
        i, j = (np.random.randint(self.margin, i_max-self.margin),
                np.random.randint(self.margin, j_max-self.margin))
        x = self.filter_bank[np.random.randint(self.n_examples)].clone().detach().cpu()

        weight = []
        # fig, axes = make_fig_axes(self.n_transforms)
        for _ in range(self.n_transforms):
            center = (i+(height-1)/2, j+(width-1)/2)
            affine_transform = self.random_affine_sampler.sample_transformation(center=center)
            x_transformed = affine_transform(x, cval=0)
            try:
                x_transformed = torch.from_numpy(x_transformed)
            except TypeError:
                pass

            w = x_transformed[:, i:i+height, j:j+width].clone().detach()
            for process in self.filter_processing:
                w = process(w)
            w = torch.unsqueeze(w, dim=0)
            weight.append(w)

        # for i, w in enumerate(weight):
        #     im = np.concatenate([ch[:,:,np.newaxis] for ch in w.numpy().reshape(3, height, width)], axis=2)
        #     axes[i].imshow(im.astype(int))
        # plt.show()

        weight = torch.cat(weight, dim=0)
        return weight, (i, j)