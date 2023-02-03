import numpy as np

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


def _sigma_mad(data):
    """Return a robust estimation of the variance.

    It assums that is a sparse vector polluted by gaussian noise.
    """
   # return np.median(np.abs(data - np.median(data)))/0.6745
    return np.median(np.abs(data))/0.6745

def _sure_est(data):
    """Return an estimation of the threshold computed using the SURE method."""
    dataf = data.flatten()
    n = dataf.size
    data_sorted = np.sort(np.abs(dataf))**2
    idx = np.arange(n-1, -1, -1)
    tmp = np.cumsum(data_sorted) + idx * data_sorted

    risk = (n - (2 * np.arange(n)) + tmp) / n
    ibest = np.argmin(risk)

    return np.sqrt(data_sorted[ibest])

def _thresh_select(data, thresh_est):
    """
    Threshold selection for denoising.

    It assumes that data has a white noise of N(0,1)
    """
    n = data.size
    universal_thr = np.sqrt(2*np.log(n))

    if thresh_est == "sure":
        thr = _sure_est(data)
    if thresh_est == "universal":
        thr = universal_thr
    if thresh_est == "hybrid-sure":
        eta = np.sum(data ** 2) /n  - 1
        if eta < (np.log2(n) ** 1.5) / np.sqrt(n):
            thr = universal_thr
        else:
            test_th = _sure_est(data)
            thr = min(test_th, universal_thr)
    return thr

def _wavelet_noise_estimate(wavelet_coefs, coeffs_shape, sigma_est):
    r"""Return an estimate of the noise variance in each band.

    Parameters
    ----------
    wavelet_bands: list
        list of array
    sigma_est: str
        Estimation method, available are "band", "level", "level-shared", "global"
    Returns
    -------
    numpy.ndarray
        Estimation of the variance for each wavelet bands.

    Notes
    -----
    This methods makes several assumptions:

     - The wavelet coefficient are ordered by scale, and the scale are ordered by size.
     - At each scale, the subbands should have the same shape.

    The variance estimation can be done:

     - On each band (eg LH, HL and HH band of each level)
     - On each level, using the HH band.
     - Only with the latest band (global)

    For the selected data band(s) the variance is estimated using the MAD estimator:

    .. math::
       \hat{\sigma} = \textrm{median}(|x|) / 0.6745

    """
    sigma_ret = np.ones(len(coeffs_shape))
    sigma_ret[0] = np.NaN
    start = 0
    stop = 0
    if sigma_est is None:
        return sigma_ret
    if sigma_est == "band":
        for i in range(1, len(coeffs_shape)):
            stop += np.prod(coeffs_shape[i])
            sigma_ret[i] = _sigma_mad(wavelet_coefs[start:stop])
            start = stop
    if sigma_est == "level":
        # use the diagonal coefficient to estimate the variance of the level.
        # it assumes that the band of the same level have the same shape.
        start = np.prod(coeffs_shape[0])
        for i, scale_shape in enumerate(np.unique(coeffs_shape[1:], axis=0)):
            scale_sz = np.prod(scale_shape)
            matched_bands = np.all(scale_shape == coeffs_shape[1:], axis=1)
            band_per_level = np.sum(matched_bands)
            start = start + scale_sz * (band_per_level-1)
            stop = start + scale_sz * band_per_level
            sigma_ret[1+i*(band_per_level):1+(i+1)*band_per_level] = _sigma_mad(wavelet_coefs[start:stop])
            start = stop
    if sigma_est == "level-shared":
        start = np.prod(coeffs_shape[0])
        for i, scale_shape in enumerate(np.unique(coeffs_shape[1:], axis=0)):
            scale_sz = np.prod(scale_shape)
            band_per_level = np.sum(scale_shape == coeffs_shape)
            stop = start + scale_sz * band_per_level
            sigma_ret[i:i+band_per_level] = _sigma_mad(wavelet_coefs[start:stop])
            start = stop
    if sigma_est == "global":
        sigma_ret *= _sigma_mad(wavelet_coefs[-np.prod(coeffs_shape[-1]):])
    sigma_ret[0] = np.NaN
    return sigma_ret

class AutoWeightedSparseThreshold(SparseThreshold):
    """Automatic Weighting of Sparse coefficient.

    This proximty automatically determines the threshold for Sparse (e.g. Wavelet based)
    coefficients.

    The weight are  computed on first call, and updated every ``update_period`` calls.
    Note that the coarse/approximation scale will not be thresholded.

    Parameters
    ----------
    coeffs_shape: list of tuple
        list of shape for the subbands.
    linear: LinearOperator
        Required for cost estimation.
    update_period: int
        Estimation of the weight update period.
    threshold_estimation: str
        threshold estimation method. Available are "sure", "hybrid-sure" and "universal"
    sigma_estimation: str
        noise std estimation method. Available are "global", "level" and "level_shared"
    thresh_type: str
        "hard" or "soft" thresholding.
    """
    def __init__(self, coeffs_shape,  linear=Identity(), update_period=0,
               sigma_estimation="global", threshold_estimation="sure", **kwargs):
        self._n_op_calls = 0
        self.cf_shape = coeffs_shape
        self._update_period = update_period


        if sigma_estimation not in ["bands", "level", "global"]:
            raise ValueError("Unsupported sigma estimation method")
        if threshold_estimation not in ["sure", "hybrid-sure", "universal"]:
            raise ValueError("Unsupported threshold estimation method.")

        self._sigma_estimation = sigma_estimation
        self._thresh_estimation = threshold_estimation


        weights_init = np.zeros(np.sum(np.prod(coeffs_shape, axis=-1)))
        super().__init__(weights=weights_init,
                         linear=linear,
                         **kwargs)

    def _auto_thresh(self, input_data):
        """Compute the best weights for the input_data."""

        # Estimate the noise std for each band.

        sigma_bands = _wavelet_noise_estimate(input_data, self.cf_shape, self._sigma_estimation)
        weights = np.zeros_like(input_data)

        # compute the threshold for each subband

        start = np.prod(self.cf_shape[0])
        stop = start
        ts = []
        for i in range(1, len(self.cf_shape)):
            stop = start + np.prod(self.cf_shape[i])
            t = sigma_bands[i] * _thresh_select(
                input_data[start:stop] / sigma_bands[i],
                self._thresh_estimation
            )
            ts.append(t)
            weights[start:stop] = t
            start = stop
        return weights

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
        if self._update_period == 0 and self._n_op_calls == 0:
            self.weights = self._auto_thresh(input_data)
        if self._update_period != 0 and self._n_op_calls % self._update_period == 0:
            self.weights = self._auto_thresh(input_data)

        self._n_op_calls += 1
        return super()._op_method(input_data, extra_factor=extra_factor)
