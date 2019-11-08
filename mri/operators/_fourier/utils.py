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


def get_stacks_fourier(kspace_loc):
    """Function that splits an incoming 3D stacked k-space samples
    into a 2D non-Cartesian plane and the vector containing the z k-space
    values of all the plane and converts to stacks of 2D. This function also
    checks for any issues of the incoming k-space pattern and if the stack
    property is not satisfied.
    Stack Property: The k-space locations originate from a stack of 2D samples.

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
    first_stack_len = np.size(np.where(kspace_loc[:, 2] == np.min(
        kspace_loc[:, 2])))
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


def gridded_inverse_fourier_transform_stack(kspace_plane_loc, z_sample_loc,
                                            kspace_data, grid, method):
    """
    This function calculates the gridded Inverse fourier transform
    from Interpolated non-Cartesian data into a cartesian grid. However,
    the IFFT is done similar to Stacked FOurier transform.

    Parameters
    ----------
    kspace_plane_loc: np.ndarray
        The N-D k_space locations of size [M, N]. These hold locations only
        in plane, extracted using get_stacks_fourier function
    z_sample_loc: np.ndarray
        This holds the z-sample locations for stacks. Again, extracted using
        get_stacks_fourier function
    kspace_data: np.ndarray
        The k-space data corresponding to kspace_plane_loc above
    grid: np.ndarray
        The Gridded matrix for which you want to calculate k_space Smaps
    method: {'linear', 'nearest', 'cubic'}, optional
        Method of interpolation for more details see scipy.interpolate.griddata
        documentation
    Returns
    -------
    np.ndarray
        The gridded inverse fourier transform of given kspace data
    """
    gridded_kspace = []
    stack_len = len(kspace_plane_loc)
    for i in range(len(z_sample_loc)):
        gridded_kspace.append(
            griddata(kspace_plane_loc,
                     kspace_data[i*stack_len:(i+1)*stack_len],
                     grid,
                     method=method,
                     fill_value=0))
    # Move the slice axis to last : Make to Nx x Ny x Nz
    gridded_kspace = np.moveaxis(np.asarray(gridded_kspace), 0, 2)
    # Transpose every image in each slice
    return np.swapaxes(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(
        gridded_kspace))), 0, 1)
