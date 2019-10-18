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
import warnings


def convert_mask_to_locations(mask):
    """ Return the converted Cartesian mask as sampling locations.

    Parameters
    ----------
    mask: np.ndarray, {0,1}
        ND matrix, not necessarly a square matrix.

    Returns
    -------
    samples_locations: np.ndarray
        samples location between [-0.5, 0.5[ of shape MxN where M is the
        number of 1 values in the mask.
    """
    locations = np.where(mask == 1)
    rslt = []
    for dim, loc in enumerate(locations):
        loc_n = loc.astype("float") / mask.shape[dim] - 0.5
        rslt.append(loc_n)

    return np.asarray(rslt).T


def convert_locations_to_mask(samples_locations, img_shape):
    """ Return the converted the sampling locations as Cartesian mask.

    Parameters
    ----------
    samples_locations: np.ndarray
        samples locations between [-0.5, 0.5[.
    img_shape: tuple of int
        shape of the desired mask, not necessarly a square matrix.

    Returns
    -------
    mask: np.ndarray, {0,1}
        2D matrix, not necessarly a square matrix.
    """
    if samples_locations.shape[-1] != len(img_shape):
        raise ValueError("Samples locations dimension doesn't correspond to ",
                         "the dimension of the image shape")
    locations = np.copy(samples_locations).astype("float")
    test = []
    locations += 0.5
    for dimension in range(len(img_shape)):
        locations[:, dimension] *= img_shape[dimension]
        if locations[:, dimension].max() >= img_shape[dimension]:
            warnings.warn("One or more samples have been found to exceed " +
                          "image dimension. They will be removed")
            locations = np.delete(locations, np.where(
                locations[:, dimension] >= img_shape[dimension]), 0)
        locations[:, dimension] = np.floor(locations[:, dimension])
        test.append(list(locations[:, dimension].astype("int")))
    mask = np.zeros(img_shape, dtype="int")
    mask[test] = 1
    return mask


def normalize_frequency_locations(samples, Kmax=None):
    """
    This function normalize the samples locations between [-0.5; 0.5[ for
    the non-cartesian case

    Parameters
    ----------
    samples: np.ndarray
        Unnormalized samples
    Kmax: int, float, array-like or None
        Maximum Frequency of the samples locations is supposed to be equal to
        base Resolution / (2* Field of View)

    Returns
    -------
    normalized_samples: np.ndarray
        Same shape as the parameters but with values between [-0.5; 0.5[
    """
    samples_locations = np.copy(samples.astype('float'))
    if Kmax is None:
        Kmax = 2*np.abs(samples_locations).max(axis=0)
    elif isinstance(Kmax, (float, int)):
        Kmax = [Kmax] * samples_locations.shape[-1]
    Kmax = np.array(Kmax)
    samples_locations /= Kmax
    if samples_locations.max() == 0.5:
        warnings.warn("Frequency equal to 0.5 will be put in -0.5")
        samples_locations[np.where(samples_locations == 0.5)] = -0.5
    return samples_locations


def generate_operators(data, wavelet_name, samples, mu=1e-06, nb_scales=4,
                       non_cartesian=False, uniform_data_shape=None,
                       gradient_space="analysis", padding_mode="zero",
                       verbose=False):
    """ Function that ease the creation of a set of common operators.

    .. note:: At the moment, supports only 2D data.

    Parameters
    ----------
    data: ndarray
        the data to reconstruct: observation are expected in Fourier space.
    wavelet_name: str | int
        if implementation is with waveletN
            the wavelet name to be used during the decomposition
        else
            implementation with waveletUD2 where the wavelet name is wavelet_id
            Refer to help of mr_transform under option '-t' to choose the right
            wavelet_id
    samples: np.ndarray
        the mask samples in the Fourier domain.
    mu: float, (defaul=1e-06) or np.ndarray
        Regularization hyper-parameter should be positif
    nb_scales: int, default 4
        the number of scales in the wavelet decomposition.
    non_cartesian: bool (optional, default False)
        if set, use the nfftw rather than the fftw. Expect an 1D input dataset.
    uniform_data_shape: uplet (optional, default None)
        the shape of the matrix containing the uniform data. Only required
        for non-cartesian reconstructions.
    gradient_space: str (optional, default 'analysis')
        the space where the gradient operator is defined: 'analysis' or
        'synthesis'
    padding_mode: str, default zero
        ways to extend the signal when computing the decomposition.
    verbose: bool, default False
        Defines verbosity for debug. If True, cost is printed at every
        iteration

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
    from mri.numerics.linear import WaveletN, WaveletUD2
    from mri.numerics.fourier import FFT2
    from mri.numerics.fourier import NFFT
    from mri.numerics.gradient import GradAnalysis2
    from mri.numerics.gradient import GradSynthesis2
    from modopt.opt.linear import Identity
    from modopt.opt.proximity import SparseThreshold

    # Check input parameters
    if gradient_space not in ("analysis", "synthesis"):
        raise ValueError("Unsupported gradient space '{0}'.".format(
            gradient_space))
    if non_cartesian and data.ndim != 1:
        raise ValueError("Expect 1D data with the non-cartesian option.")
    elif non_cartesian and uniform_data_shape is None:
        raise ValueError("Need to set the 'uniform_data_shape' parameter with "
                         "the non-cartesian option.")
    elif not non_cartesian and data.ndim != 2:
        raise ValueError("At the moment, this functuion only supports 2D "
                         "data.")
    # Define the linear/fourier operators
    try:
        linear_op = WaveletN(
            nb_scale=nb_scales,
            wavelet_name=wavelet_name,
            padding_mode=padding_mode)
    except:
        # For Undecimated wavelets, the wavelet_name is wavelet_id
        linear_op = WaveletUD2(wavelet_id=wavelet_name,
                               nb_scale=nb_scales)
    if non_cartesian:
        fourier_op = NFFT(
            samples=samples,
            shape=uniform_data_shape)
    else:
        fourier_op = FFT2(
            samples=samples,
            shape=data.shape)

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
            fourier_op=fourier_op)
        prox_op = SparseThreshold(Identity(), mu, thresh_type="soft")
    else:
        gradient_op = GradAnalysis2(
            data=data,
            fourier_op=fourier_op)
        prox_op = SparseThreshold(linear_op, mu, thresh_type="soft")

    # Define the cost function
    # TODO need to have multiple cost functions with a parameter
    cost_op = GenericCost(
        gradient_op=gradient_op,
        prox_op=prox_op,
        verbose=verbose)
    return gradient_op, linear_op, prox_op, cost_op
