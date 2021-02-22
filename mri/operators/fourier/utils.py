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
from scipy.interpolate import griddata, RegularGridInterpolator


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
        Kmax = np.abs(samples_locations).max(axis=0)
    elif isinstance(Kmax, (float, int)):
        Kmax = [Kmax] * samples_locations.shape[-1]
    Kmax = np.array(Kmax)
    samples_locations /= (2 * Kmax)
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
        # We use np.isclose rather than '==' as the actual z_loc comes
        # from scanner binary file that has been limited to floats
        idx_mask_z = np.asarray([
            np.where(np.isclose(z_loc, full_stack_z_loc))[0][0]
            for z_loc in sampled_stack_z_loc
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
    We expect the kspace data to be limited to a grid on z, we calculate
    the inverse fourier transform by-
    1) Grid data in each plane (for all points in a plane)
    2) Interpolate data along z, if we have undersampled data along z
    3) Apply an IFFT on the 3D data that was gridded and interpolated in z.

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
    # Check if we have undersampled in Z direction, in which case,
    # we need to interpolate values along z to get a good reconstruction.
    if len(idx_mask_z) < volume_shape[2]:
        # Interpolate along z direction
        grid_loc = [
            np.linspace(-0.5, 0.5, volume_shape[i], endpoint=False)
            for i in range(3)
        ]
        interp_z = RegularGridInterpolator(
            (*grid_loc[0:2], grid_loc[2][idx_mask_z]),
            gridded_kspace[:, :, idx_mask_z],
            bounds_error=False,
            fill_value=None,
        )
        unsampled_z = list(
            set(np.arange(volume_shape[2])) - set(idx_mask_z)
        )
        mask = np.zeros(volume_shape)
        mask[:, :, unsampled_z] = 1
        loc = convert_mask_to_locations(mask)
        gridded_kspace[:, :, unsampled_z] = np.reshape(
            interp_z(loc),
            (*volume_shape[0:2], len(unsampled_z)),
        )
    # Transpose every image in each slice
    return np.swapaxes(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(
        gridded_kspace))), 0, 1)


def check_if_fourier_op_uses_sense(fourier_op):
    """Utils function to check if fourier operator uses SENSE recon

    Parameters
    ----------

    fourier_op: object of class FFT, NonCartesianFFT or Stacked3DNFFT in
    mri.operators
        the fourier operator for which we want to check if SENSE is
        supported

    Returns
    -------
    bool
        True if SENSE recon is being used
    """
    from .non_cartesian import NonCartesianFFT, gpuNUFFT
    if isinstance(fourier_op, NonCartesianFFT) and \
            isinstance(fourier_op.impl, gpuNUFFT):
        return fourier_op.impl.uses_sense
    else:
        return False


def estimate_density_compensation(kspace_loc, volume_shape, num_iterations=10):
    """ Utils function to obtain the density compensator for a
    given set of kspace locations.

    Parameters
    ----------
    kspace_loc: np.ndarray
        the kspace locations
    volume_shape: np.ndarray
        the volume shape
    num_iterations: int default 10
        the number of iterations for density estimation
    """
    from .non_cartesian import NonCartesianFFT
    from .non_cartesian import gpunufft_available
    if gpunufft_available is False:
        raise ValueError("gpuNUFFT is not available, cannot "
                         "estimate the density compensation")
    grid_op = NonCartesianFFT(
        samples=kspace_loc,
        shape=volume_shape,
        implementation='gpuNUFFT',
        osf=1,
    )
    density_comp = np.ones(kspace_loc.shape[0])
    for _ in range(num_iterations):
        density_comp = (
                density_comp /
                np.abs(grid_op.op(grid_op.adj_op(density_comp, True), True))
        )
    return density_comp
