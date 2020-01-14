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
import warnings

# Third party import
import numpy as np
import scipy.fftpack as pfft
from scipy.interpolate import griddata


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


def get_stacks_fourier(kspace_loc, volume_shape):
    """Function that splits an incoming 3D stacked k-space samples
    into a 2D non-Cartesian plane and the vector containing the z k-space
    values of the stacks acquiered and converts to stacks of 2D.
    This function also checks for any issues of the incoming k-space
    pattern and if the stack property is not satisfied.
    Stack Property: The k-space locations originate from a stack of 2D samples.

    Parameters
    ----------
    kspace_loc: np.ndarray
        Acquired 3D k-space locations : stacks of same non-Cartesian samples,
        while Cartesian under-sampling on the stacks direction.
    volume_shape: tuple
        Reconstructed volume shape
    Returns
    ----------
    kspace_plane_loc: np.ndarray
        A 2D array of samples which when stacked gives the 3D samples
    z_sample_loc: np.ndarray
        A 1D array of z-sample locations
    sort_pos: np.ndarray
        The sorting positions for opertor and inverse for incoming data
    idx_mask_z: np.ndarray
        contains the indices of the acquired Fourier planes (z direction)
    """
    # Sort the incoming data based on Z, Y then X coordinates
    # This is done for easier stacking
    sort_pos = np.lexsort(tuple(kspace_loc[:, i]
                                for i in np.arange(3)))
    kspace_loc = kspace_loc[sort_pos]

    # Find the mask used to sample stacks in z direction
    full_stack_z_loc = convert_mask_to_locations(
        np.ones(volume_shape[2]),
    )[:, 0]
    sampled_stack_z_loc = np.unique(kspace_loc[:, 2])

    try:
        idx_mask_z = np.asarray([
            np.where(x == full_stack_z_loc)[0][0] for x in sampled_stack_z_loc
        ])
    except IndexError:
        raise ValueError('The input must be a stack of 2D k-Space data')

    first_stack_len = np.size(np.where(
        kspace_loc[:, 2] == np.min(kspace_loc[:, 2])))
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
    kspace_plane_loc = stacked[0, :, 0:2]
    z_sample_loc = stacked[:, 0, 2]
    z_sample_loc = z_sample_loc[:, np.newaxis]
    return kspace_plane_loc, z_sample_loc, sort_pos, idx_mask_z


def gridded_inverse_fourier_transform_nd(kspace_loc,
                                         kspace_data, grid, method):
    """
    This function calculates the gridded Inverse fourier transform
    from Interpolated non-Cartesian data into a cartesian grid

    Parameters
    ----------
    kspace_loc: np.ndarray
        The N-D k_space locations of size [M, N]
    kspace_data: np.ndarray
        The k-space data corresponding to k-space_loc above
    grid: np.ndarray
        The Gridded matrix for which you want to calculate k_space Smaps
    method: {'linear', 'nearest', 'cubic'}
        Method of interpolation for more details see scipy.interpolate.griddata
        documentation

    Returns
    -------
    np.ndarray
        The gridded inverse fourier transform of given kspace data
    """
    gridded_kspace = griddata(kspace_loc,
                              kspace_data,
                              grid,
                              method=method,
                              fill_value=0)
    return np.swapaxes(pfft.fftshift(
        pfft.ifftn(pfft.ifftshift(gridded_kspace))), 1, 0)


def gridded_inverse_fourier_transform_stack(kspace_data_sorted,
                                            kspace_plane_loc,
                                            idx_mask_z,
                                            grid,
                                            volume_shape,
                                            method):
    """
    This function calculates the gridded Inverse fourier transform
    from Interpolated non-Cartesian data into a cartesian grid. However,
    the IFFT is done similar to Stacked Fourier transform.
    Parameters
    ----------
    kspace_data_sorted: np.ndarray
        The sorted k-space data corresponding to kspace_plane_loc above
    kspace_plane_loc: np.ndarray
        The N-D k_space locations of size [M, N]. These hold locations only
        in plane, extracted using get_stacks_fourier function
    idx_mask_z: np.ndarray
        contains the indices of the acquired Fourier plane. Extracted using
        get_stacks_fourier function
    grid: tuple
        The Gridded matrix for which you want to calculate k_space Smaps.
        Should be given as a tuple of ndarray
    volume_shape: tuple
        Reconstructed volume shape
    method: {'linear', 'nearest', 'cubic'}, optional
        Method of interpolation for more details see scipy.interpolate.griddata
        documentation
    Returns
    -------
    np.ndarray
        The gridded inverse fourier transform of given kspace data
    """
    gridded_kspace = np.zeros(volume_shape, dtype=kspace_data_sorted.dtype)
    stack_len = len(kspace_plane_loc)
    for i, idx_z in enumerate(idx_mask_z):
        gridded_kspace[:, :, idx_z] = griddata(
            kspace_plane_loc,
            kspace_data_sorted[i*stack_len:(i+1)*stack_len],
            grid,
            method=method,
            fill_value=0,
        )
    # Transpose every image in each slice
    return np.swapaxes(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(
        gridded_kspace))), 0, 1)
