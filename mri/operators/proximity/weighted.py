import numpy as np

from modopt.opt.proximity import SparseThreshold


class WeightedSparseThreshold(SparseThreshold):
    """This is a weighted version of `SparseThreshold` in ModOpt.
    When chosen `scale_based`, this applied weighted proximity as :
        W * P^i where i is the scale
    Also, custom weights can be defined.
    Note that the weights on coarse scale is always set to 0

    Parameters
    ----------
    weights : numpy.ndarray
        Input array of weights or a tuple holding base weight W and power P
    coeffs_shape: tuple
        The shape of linear coefficients
    weight_type : string 'custom' | 'scale_based', default 'scale_based'
        Mode of operation of proximity:
        custom      -> custom array of weights
        scale_based -> weights applied per scale
    See Also
    --------
    SparseThreshold : parent class
    """
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
        """`mu` is a parameter which controls the weights"""
        return self.weights

    @mu.setter
    def mu(self, w):
        """Update `mu`, based on `coeffs_shape` and `weight_type`"""
        weights_init = np.zeros(np.sum(np.prod(self.coeffs_shape, axis=-1)))
        start = 0
        if self.weight_type == 'scale_based':
            if isinstance(w, (float, int, np.float64)):
                base_weight = w
                power = 1
            else:
                base_weight, power = w
            for i, scale_shape in enumerate(np.unique(self.coeffs_shape, axis=0)):
                scale_size = np.prod(scale_shape)
                stop = start + scale_size * np.sum(scale_shape == self.coeffs_shape)
                weights_init[start:stop] = base_weight * (power ** i)
                start = stop
        elif self.weight_type == 'custom':
            if isinstance(w, (float, int, np.float64)):
                w = w * np.ones(weights_init.shape[0])
            weights_init = w
        weights_init[:np.prod(self.coeffs_shape[0])] = 0
        self.weights = weights_init
