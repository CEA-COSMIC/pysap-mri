import numpy as np
import scipy as sp

from modopt.opt.proximity import SparseThreshold
from modopt.opt.linear import Identity


class WeightedSparseThreshold(SparseThreshold):
    """This is a weighted version of `SparseThreshold` in ModOpt.
    When chosen `scale_based`, it allows the users to specify an array of
    weights W[i] and each weight is assigen to respective scale `i`.
    Also, custom weights can be defined.
    Note that the weights on coarse scale is always set to 0

    Parameters
    ----------
    weights : numpy.ndarray
        Input array of weights or a tuple holding base weight W and power P
    coeffs_shape: tuple
        The shape of linear coefficients
    weight_type : string 'custom' | 'scale_based' | 'custom_scale',
        default 'scale_based'
        Mode of operation of proximity:
        custom       -> custom array of weights
        scale_based -> custom weights applied per scale
    zero_weight_coarse: bool, default True
    linear: object, default `Identity()`
        Linear operator, to be used in cost function evaluation

    See Also
    --------
    SparseThreshold : parent class
    """
    def __init__(self, weights, coeffs_shape, weight_type='scale_based',
                 zero_weight_coarse=True, linear=Identity(), **kwargs):
        self.cf_shape = coeffs_shape
        self.weight_type = weight_type
        available_weight_type = ('scale_based', 'custom')
        if self.weight_type not in available_weight_type:
            raise ValueError('Weight type must be one of ' +
                             ' '.join(available_weight_type))
        self.zero_weight_coarse = zero_weight_coarse
        self.mu = weights
        super(WeightedSparseThreshold, self).__init__(
            weights=self.mu,
            linear=linear,
            **kwargs
        )

    @property
    def mu(self):
        """`mu` is the weights used for thresholding"""
        return self.weights

    @mu.setter
    def mu(self, w):
        """Update `mu`, based on `coeffs_shape` and `weight_type`"""
        weights_init = np.zeros(np.sum(np.prod(self.cf_shape, axis=-1)))
        start = 0
        if self.weight_type == 'scale_based':
            scale_shapes = np.unique(self.cf_shape, axis=0)
            num_scales = len(scale_shapes)
            if isinstance(w, (float, int, np.float64)):
                weights = w * np.ones(num_scales)
            else:
                if len(w) != num_scales:
                    raise ValueError('The number of weights dont match '
                                     'the number of scales')
                weights = w
            for i, scale_shape in enumerate(np.unique(self.cf_shape, axis=0)):
                scale_sz = np.prod(scale_shape)
                stop = start + scale_sz * np.sum(scale_shape == self.cf_shape)
                weights_init[start:stop] = weights[i]
                start = stop
        elif self.weight_type == 'custom':
            if isinstance(w, (float, int, np.float64)):
                w = w * np.ones(weights_init.shape[0])
            weights_init = w
        if self.zero_weight_coarse:
            weights_init[:np.prod(self.cf_shape[0])] = 0
        self.weights = weights_init


class AutoWeightedSparseThreshold(WeightedSparseThreshold):
    """This WeightedSparseThreshold uses the universal threshold rules to """
    def __init__(self, coeffs_shape,  linear=Identity(), update_period=0,
               sigma_estimation="global", threshold_estimation="sure", **kwargs):
        self._n_op_calls = 0
        self._sigma_estimation = sigma_estimation
        self._update_period = update_period

        self._thresh_estimation = threshold_estimation

        weights_init = np.zeros(np.sum(np.prod(self.cf_shape, axis=-1)))
        super().__init__(weights=weights_init,
                         weight_type="scale_based",
                         **kwargs)

    def _auto_thresh_scale(self, input_data, sigma=None):
        """Determine a threshold value adapted to denoise the data of a specific scale.

        Parameters
        ----------
        input_data: numpy.ndarray
            data that should be thresholded.
        sigma: float
            Estimation of the noise standard deviation.
        Returns
        -------
        float
            The estimated threshold.

        Raises
        ------
            ValueError is method is not supported.
        Notes
        -----
        The choice of the threshold makes the assumptions of a white additive gaussian noise.
        """

        #tmp = np.sort(input_data.flatten())
        tmp = input_data.flatten()
        # use the robust estimator to estimate the noise variance.
        med = np.median(tmp)
        if sigma is None:
            sigma = np.median(np.abs(tmp-med)) / 0.6745
        N = len(input_data)
        j = np.log2(N)

        uni_threshold = np.sqrt(2*np.log(N))

        if self._thresh_estimation == "universal":
            return sigma * uni_threshold, sigma
        elif self._thresh_estimation == "sure":
            tmp = tmp **2
            eps2 = (sigma ** 2) /N
            # TODO: optimize the estimation
            def _sure(t):
                e2t2 = eps2 * (t**2)
                return (N * eps2
                   + np.sum(np.minimum(tmp, e2t2))
                   - 2*eps2*np.sum(tmp <= e2t2)
                )

            thresh = sp.optimize.minimize_scalar(
                _sure,
                method="bounded",
                bounds=[0, uni_threshold]).x
            sj2 = np.sum(tmp/eps2 - 1)/N
            if sj2 >= 3*j/np.sqrt(2*N):
                return thresh, sigma
            else:
                return uni_threshold, sigma

        else:
            raise ValueError("Unknown method name")

    def _auto_thresh(self, input_data):
        """Determines the threshold for every scale (except the coarse one) using the provided method.

        Parameters
        ----------
        input_data: list of numpy.ndarray

        Returns
        -------
        thresh_list: list
            list of threshold for every scale
        """
        sigma = None
        thresh_list = []
        for band_idx in range(len(input_data)-1, 1, -1):
            thresh_value, sigma_est = self._auto_thresh_scale(input_data[band_idx], sigma=sigma)
            if self._sigma_estimation == "global" and band_idx == len(input_data)-1:
                sigma = sigma_est
            thresh_list.append(thresh_value)
        return thresh_list


    def _op_method(self, input_data, extra_factor=1.0):
        """Operator.

        This method returns the input data thresholded by the weights.
        The weights are computed using the universal threshold rule.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor (default is ``1.0``)

        Returns
        -------
        numpy.ndarray
            Thresholded data

        """
        if (self._update_period == 0 and self._n_op_calls == 0) or (self._n_op_calls % self._update_period == 0) :
            self.mu , sigma = self._auto_thresh(input_data)
        self._n_op_calls += 1
        return super()._op_method(input_data, extra_factor=extra_factor)
