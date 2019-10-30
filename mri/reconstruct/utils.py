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
    from mri.numerics.linear import WaveletN, WaveletUD2
    from mri.numerics.fourier import FFT2, NonCartesianFFT, Stacked3DNFFT
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
    elif fourier_type == 'cartesian' and data.ndim != 2:
        raise ValueError("At the moment, this functuion only supports 2D "
                         "data.")
    elif fourier_type == 'stack' and len(uniform_data_shape) == 3 and \
            samples.shape[-1] == 3:
        raise ValueError("Stack version can only be used in 3D.")
    # Define the linear/fourier operators
    if fourier_type == 'non-cartesian':
        fourier_op = NonCartesianFFT(
            samples=samples,
            shape=uniform_data_shape,
            implementation=nfft_implementation)
    elif fourier_type == 'cartesian':
        fourier_op = FFT2(
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


def get_stacks_fourier(kspace_loc):
    """Function that splits an incoming 3D stacked k-space samples
    into a 2D non-Cartesian plane and the vector containing the z k-space
    values of all the plane and converts to stacks of 2D. This function also
    checks for any issues of the incoming k-space pattern and if the stack
    property is not satisfied.
    Stack Property:
        The k-space locations originate from a stack of 2D samples
    Parameters
    ----------
    ksapce_plane_loc: np.ndarray
        the mask samples in the 3D Fourier domain.

    Returns
    ----------
    ksapce_plane_loc: np.ndarray
        A 2D array of samples which when stacked gives the 3D samples
    z_sample_loc: np.ndarray
        A 1D array of z-sample locations
    sort_pos: np.ndarray
        The sorting positions for opertor and inverse for incoming data
    """
    # Sort the incoming data based on Z, Y then X coordinates
    # This is done for easier stacking
    sort_pos = np.lexsort(tuple(kspace_loc[:, i]
                                for i in np.arange(3)))
    kspace_loc = kspace_loc[sort_pos]
    first_stack_len = np.size(np.where(kspace_loc[:, 2]
                                       == np.min(kspace_loc[:, 2])))
    acq_num_slices = int(len(kspace_loc) / first_stack_len)
    stacked = np.reshape(kspace_loc, (acq_num_slices,
                                      first_stack_len, 3))
    z_expected_stacked = np.reshape(np.repeat(stacked[:, 0, 2],
                                              first_stack_len),
                                    (acq_num_slices,
                                     first_stack_len))
    if np.mod(len(kspace_loc), first_stack_len) \
            or not np.all(stacked[:, :, 0:2] == stacked[0, :, 0:2]) \
            or not np.all(stacked[:, :, 2] == z_expected_stacked):
        raise ValueError('The input must be a stack of 2D k-Space data')
    ksapce_plane_loc = stacked[0, :, 0:2]
    z_sample_loc = stacked[:, 0, 2]
    z_sample_loc = z_sample_loc[:, np.newaxis]
    return ksapce_plane_loc, z_sample_loc, sort_pos
