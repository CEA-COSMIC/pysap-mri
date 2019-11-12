# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
"""
Common tools for MRI image reconstruction.
"""


# System import
import numpy as np


def generate_operators(data, wavelet_name, samples, mu=1e-06, nb_scales=4,
                       fourier_type='cartesian', uniform_data_shape=None,
                       gradient_space="analysis", padding_mode="zero",
                       nfft_implementation='cpu', lips_calc_max_iter=5,
                       verbose=False):
    """ Function that ease the creation of a set of common operators.
    .. note:: At the moment, supports only 2D data.
    Parameters
    ----------
    data: ndarray
        the data to reconstruct: observation are expected in Fourier space.
    wavelet_name: str | int
        if implementation is with waveletN the wavelet name to be used during
        the decomposition, else implementation with waveletUD2 where the
        wavelet name is wavelet_id Refer to help of mr_transform under option
        '-t' to choose the right wavelet_id.
    samples: np.ndarray
        the mask samples in the Fourier domain.
    mu: float, (defaul=1e-06) or np.ndarray
        Regularization hyper-parameter should be positif
    nb_scales: int, default 4
        the number of scales in the wavelet decomposition.
    fourier_type: str (optional, default 'cartesian')
        type of fourier operator : 'cartesian' | 'non-cartesian' | 'stack'
    uniform_data_shape: uplet (optional, default None)
        the shape of the matrix containing the uniform data. Only required
        for non-cartesian reconstructions.
    gradient_space: str (optional, default 'analysis')
        the space where the gradient operator is defined: 'analysis' or
        'synthesis'
    padding_mode: str, default zero
        ways to extend the signal when computing the decomposition.
    nfft_implementation: str, default 'cpu'
        way to implement NFFT : 'cpu' | 'cuda' | 'opencl'
    verbose: bool, default False
        Defines verbosity for debug. If True, cost is printed at every
        iteration
    lips_calc_max_iter: int, default 10
        Defines the maximum number of iterations to calculate the lipchitz
        constant
    Returns
    -------
    gradient_op: instance of class GradBase
        the gradient operator.
    linear_op: instance of LinearBase
        the linear operator: seek the sparsity, ie. a wavelet transform.
    prox_op: instance of ProximityParent
        the proximal operator.
    cost_op: instance of costObj
        the cost function used to check for convergence during the
        optimization.
    """
    # Local imports
    from mri.numerics.cost import GenericCost
    from mri.operators import WaveletN, WaveletUD2
    from mri.operators import FFT, NonCartesianFFT, Stacked3DNFFT
    from mri.numerics.gradient import GradAnalysis2
    from mri.numerics.gradient import GradSynthesis2
    from modopt.opt.linear import Identity
    from modopt.opt.proximity import SparseThreshold

    # Check input parameters
    if gradient_space not in ("analysis", "synthesis"):
        raise ValueError("Unsupported gradient space '{0}'.".format(
            gradient_space))
    if fourier_type == 'non-cartesian' and data.ndim != 1:
        raise ValueError("Expect 1D data with the non-cartesian option.")
    elif fourier_type == 'non-cartesian' and uniform_data_shape is None:
        raise ValueError("Need to set the 'uniform_data_shape' parameter with "
                         "the non-cartesian option.")
    elif fourier_type == 'stack' and len(uniform_data_shape) == 3 and \
            samples.shape[-1] != 3:
        raise ValueError("Stack version can only be used in 3D.")

    # Define the linear/fourier operators
    if fourier_type == 'non-cartesian':
        fourier_op = NonCartesianFFT(
            samples=samples,
            shape=uniform_data_shape,
            implementation=nfft_implementation)
    elif fourier_type == 'cartesian':
        fourier_op = FFT(
            samples=samples,
            shape=data.shape)
    elif fourier_type == 'stack':
        fourier_op = Stacked3DNFFT(kspace_loc=samples,
                                   shape=uniform_data_shape,
                                   implementation=nfft_implementation)
    else:
        raise ValueError('The value of fourier_type must be "cartesian" | '
                         '"non-cartesian" | "stack"')
    try:
        linear_op = WaveletN(
            nb_scale=nb_scales,
            wavelet_name=wavelet_name,
            padding_mode=padding_mode,
            dim=len(fourier_op.shape))
    except ValueError:
        # For Undecimated wavelets, the wavelet_name is wavelet_id
        linear_op = WaveletUD2(wavelet_id=wavelet_name,
                               nb_scale=nb_scales)
    # Check SparseThreshold hyper-parameter
    if type(mu) is np.ndarray:
        x_init = linear_op.op(np.zeros(fourier_op.shape))
        if mu.shape != x_init.shape:
            raise ValueError("The mu shape should be equal to: {0}".format(
                x_init.shape))
        elif any(mu_values < 0 for mu_values in mu):
            raise ValueError("The mu hyper-parameter vector"
                             " should be positive")
    elif mu < 0:
        raise ValueError("The mu hyper-parameter should be positive")

    # Define the gradient and proximity operators
    if gradient_space == "synthesis":
        gradient_op = GradSynthesis2(
            data=data,
            linear_op=linear_op,
            fourier_op=fourier_op,
            max_iter_spec_rad=lips_calc_max_iter)
        prox_op = SparseThreshold(Identity(), mu, thresh_type="soft")
    else:
        gradient_op = GradAnalysis2(
            data=data,
            fourier_op=fourier_op,
            max_iter_spec_rad=lips_calc_max_iter)
        prox_op = SparseThreshold(linear_op, mu, thresh_type="soft")
    # Define the cost function
    # TODO need to have multiple cost functions with a parameter
    cost_op = GenericCost(
        gradient_op=gradient_op,
        prox_op=prox_op,
        verbose=verbose)
    return gradient_op, linear_op, prox_op, cost_op
