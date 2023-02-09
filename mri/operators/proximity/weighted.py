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


def _sigma_mad(data, centered=True):
    """Return a robust estimation of the standard deviation.

    The standard deviation is computed using the following estimator, based on the
    Median Absolute deviation of the data [#]_
    .. math::
        \hat{\sigma} = \frac{MAD}{\sqrt{2}\textrm{erf}^{-1}(1/2)}

    Parameters
    ----------
    data: numpy.ndarray
        the data on which the standard deviation will be estimated.
    centered: bool, default True.
        If true the median of the is assummed to be 0.
    Returns
    -------
    float:
        The estimation of the standard deviation.

    References
    ----------
    .. [#] https://en.m.wikipedia.org/wiki/Median_absolute_deviation
    """
    if centered:
        return np.median(np.abs(data[:]))/0.6745
    return np.median(np.abs(data[:] - np.median(data[:])))/0.6745

def _sure_est(data):
    """Return an estimation of the threshold computed using the SURE method.

    The computation of the estimator is based on the formulation of `cite:donoho1994`
    and the efficient implementation of [#]_

    Parameters
    ----------
    data: numpy.array
        Noisy Data with unit standard deviation.
    Returns
    -------
    float
        Value of the threshold minimizing the SURE estimator.

    References
    ----------
    .. [#] https://pyyawt.readthedocs.io/_modules/pyyawt/denoising.html#ValSUREThresh
    """
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
    Threshold selection for denoising, implementing the methods proposed in `cite:donoho1994`

    Parameters
    ----------
    data: numpy.ndarray
        Noisy data on which a threshold will be estimated. It should only be corrupted by a
        standard gaussian white noise N(0,1).
    thresh_est: str
        threshold estimation method. Available are "sure", "universal", "hybrid-sure".
    Returns
    -------
    float:
        the threshold for the data provided.
    """
    n = data.size
    universal_thr = np.sqrt(2*np.log(n))

    if thresh_est == "sure":
        thr = _sure_est(data)
    elif thresh_est == "universal":
        thr = universal_thr
    elif thresh_est == "hybrid-sure":
        eta = np.sum(data ** 2) /n  - 1
        if eta < (np.log2(n) ** 1.5) / np.sqrt(n):
            thr = universal_thr
        else:
            test_th = _sure_est(data)
            thr = min(test_th, universal_thr)
    else:
        raise ValueError(
            "Unsupported threshold method."
            "Available are 'sure', 'universal' and 'hybrid-sure'"
        )
    return thr

def wavelet_noise_estimate(wavelet_coeffs, coeffs_shape, sigma_est):
    r"""Return an estimate of the noise standard deviation in each subband.

    Parameters
    ----------
    wavelet_coeffs: numpy.ndarray
        flatten array of wavelet coefficients, typically returned by ``WaveletN.op``
    coeffs_shape:
        list of tuple representing the shape of each subband.
        Typically accessible by WaveletN.coeffs_shape
    sigma_est: str
        Estimation method, available are "band", "scale", and "global"
    Returns
    -------
    numpy.ndarray
        Estimation of the variance for each wavelet subband.

    Notes
    -----
    This methods makes several assumptions:

     - The wavelet coefficients are ordered by scale, and the scales are ordered by size.
     - At each scale, the subbands should have the same shape.

    The variance estimation is either performed:

     - On each subband (``sigma_est = "band"``)
     - On each scale, using the detailled HH subband. (``sigma_est = "scale"``)
     - Only with the largest, most detailled HH band (``sigma_est = "global"``)

    See Also
    --------
    _sigma_mad: function estimating the standard deviation.
    """
    sigma_ret = np.ones(len(coeffs_shape))
    sigma_ret[0] = np.NaN
    start = 0
    stop = 0
    if sigma_est is None:
        return sigma_ret
    elif sigma_est == "band":
        for i in range(1, len(coeffs_shape)):
            stop += np.prod(coeffs_shape[i])
            sigma_ret[i] = _sigma_mad(wavelet_coeffs[start:stop])
            start = stop
    elif sigma_est == "scale":
        # use the diagonal coefficients subband to estimate the variance of the scale.
        # it assumes that the band of the same scale have the same shape.
        start = np.prod(coeffs_shape[0])
        for i, scale_shape in enumerate(np.unique(coeffs_shape[1:], axis=0)):
            scale_sz = np.prod(scale_shape)
            matched_bands = np.all(scale_shape == coeffs_shape[1:], axis=1)
            bpl = np.sum(matched_bands)
            start = start + scale_sz * (bpl-1)
            stop = start + scale_sz * bpl
            sigma_ret[1+i*(bpl):1+(i+1)*bpl] = _sigma_mad(wavelet_coeffs[start:stop])
            start = stop
    elif sigma_est == "global":
        sigma_ret *= _sigma_mad(wavelet_coeffs[-np.prod(coeffs_shape[-1]):])
    sigma_ret[0] = np.NaN
    return sigma_ret


def wavelet_threshold_estimate(
        wavelet_coeffs,
        coeffs_shape,
        thresh_range="global",
        sigma_range="global",
        thresh_estimation="hybrid-sure"
):
    """Estimate wavelet coefficient thresholds.

    Notes that no threshold will be estimate for the coarse scale.
    Parameters
    ----------
    wavelet_coeffs: numpy.ndarray
        flatten array of wavelet coefficient, typically returned by ``WaveletN.op``
    coeffs_shape: list
        List of tuple representing the shape of each subbands.
        Typically accessible by WaveletN.coeffs_shape
    thresh_range: str. default "global"
        Defines on which data range to estimate thresholds.
        Either "band", "scale", or "global"
    sigma_range: str, default "global"
        Defines on which data range to estimate thresholds.
        Either "band", "scale", or "global"
    thresh_estimation: str, default "hybrid-sure"
        Name of the threshold estimation method.
        Available are "sure", "hybrid-sure", "universal"

    Returns
    -------
    numpy.ndarray
        array of threshold for each wavelet coefficient.
    """

    weights = np.ones(wavelet_coeffs.shape)
    weights[:np.prod(coeffs_shape[0])] = 0

    # Estimate the noise std on the specific range.

    sigma_bands = wavelet_noise_estimate(wavelet_coeffs, coeffs_shape, sigma_range)

    # compute the threshold on each specific range.

    start = np.prod(coeffs_shape[0])
    stop = start
    ts = []
    if thresh_range == "global":
        weights =sigma_bands[-1] * _thresh_select(
            wavelet_coeffs[-np.prod(coeffs_shape[-1]):] / sigma_bands[-1],
            thresh_estimation
        )
    elif thresh_range == "band":
        for i in range(1, len(coeffs_shape)):
            stop = start + np.prod(coeffs_shape[i])
            t = sigma_bands[i] * _thresh_select(
                wavelet_coeffs[start:stop] / sigma_bands[i],
                thresh_estimation
            )
            ts.append(t)
            weights[start:stop] = t
            start = stop
    elif thresh_range == "scale":
        start = np.prod(coeffs_shape[0])
        start_hh = start
        for i, scale_shape in enumerate(np.unique(coeffs_shape[1:], axis=0)):
            scale_sz = np.prod(scale_shape)
            matched_bands = np.all(scale_shape == coeffs_shape[1:], axis=1)
            band_per_scale = np.sum(matched_bands)
            start_hh = start + scale_sz * (band_per_scale-1)
            stop = start + scale_sz * band_per_scale
            t = sigma_bands[i+1] * _thresh_select(
                wavelet_coeffs[start_hh:stop] / sigma_bands[i+1],
                thresh_estimation
            )
            ts.append(t)
            weights[start:stop] = t
            start = stop
    return weights



class AutoWeightedSparseThreshold(SparseThreshold):
    """Automatic Weighting of sparse coefficients.

    This proximty automatically determines the threshold for Sparse (e.g. Wavelet based)
    coefficients.

    The weight are computed on first call, and updated on every ``update_period`` call.
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
        noise std estimation method. Available are "global", "scale" and "band"
    thresh_type: str
        "hard" or "soft" thresholding.
    """
    def __init__(self, coeffs_shape,  linear=Identity(), update_period=0,
               sigma_range="global",
               thresh_range="global",
               threshold_estimation="sure",
               threshold_scaler=1.0,
               **kwargs):
        self._n_op_calls = 0
        self.cf_shape = coeffs_shape
        self._update_period = update_period


        if thresh_range not in ["bands", "scale", "global"]:
            raise ValueError("Unsupported threshold range.")
        if sigma_range not in ["bands", "scale", "global"]:
            raise ValueError("Unsupported sigma estimation method.")
        if threshold_estimation not in ["sure", "hybrid-sure", "universal", "bayes"]:
            raise ValueError("Unsupported threshold estimation method.")

        self._sigma_range = sigma_range
        self._thresh_range = thresh_range
        self._thresh_estimation = threshold_estimation
        self._thresh_scale = threshold_scaler


        weights_init = np.zeros(np.sum(np.prod(coeffs_shape, axis=-1)))
        super().__init__(weights=weights_init,
                         linear=linear,
                         **kwargs)

    def _auto_thresh(self, input_data):
        """Compute the best weights for the input_data.

        Parameters
        ----------
        input_data: numpy.ndarray
            Array of sparse coefficient.
        See Also
        --------
        wavelet_threshold_estimate
        """
        weights = wavelet_threshold_estimate(
            input_data,
            self.cf_shape,
            thresh_range=self._thresh_range,
            sigma_range=self._sigma_range,
            thresh_estimation=self._thresh_estimation,
        )
        if callable(self._thresh_scale):
            weights = self._thresh_scale(weights, self._n_op_calls)
        else:
            weights *= self._thresh_scale
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
