import numpy as np

from modopt.opt.proximity import SparseThreshold


class WeightedSparseThreshold(SparseThreshold):
    def __init__(self, weights, coeffs_shape, weight_type='scale_based', **kwargs):
        self.coeffs_shape = coeffs_shape
        self.weight_type = weight_type
        self.mu = weights
        super(WeightedSparseThreshold, self).__init__(
            weights=self.mu,
            **kwargs
        )

    @property
    def mu(self):
        return self.weights

    @mu.setter
    def mu(self, w):
        weights_init = np.zeros(np.sum(np.prod(self.coeffs_shape, axis=-1)))
        start = 0
        if self.weight_type == 'scale_based':
            if type(w) == float or type(w) == int or len(w) == 1 or w.size == 1:
                base_weight = w
                power = 1
            else:
                base_weight, power = w
            for i, scale_shape in enumerate(np.unique(self.coeffs_shape, axis=0)):
                scale_size = np.prod(scale_shape)
                stop = start + scale_size * np.sum(scale_shape == self.coeffs_shape)
                weights_init[start:stop] = base_weight * (power ** (i-1))
                start = stop
        elif self.weight_type == 'custom':
            if type(w) == float or type(w) == int or len(w) == 1 or w.size == 1:
                w = w * np.ones(weights_init.shape[0])
            weights_init = w
        weights_init[:np.prod(self.coeffs_shape[0])] = 0
        self.weights = weights_init
