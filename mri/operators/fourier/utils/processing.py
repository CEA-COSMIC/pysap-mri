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

from mrinufft import get_density


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
    This function normalizes the sample locations between [-0.5; 0.5[ for
    the non-Cartesian case.

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
    if np.abs(samples_locations).max() >= 0.5:
        warnings.warn("Frequencies outside the 0.5 limit.")
    return samples_locations


def discard_frequency_outliers(kspace_loc, kspace_data):
    """
    This function discards the samples outside [-0.5; 0.5[ for
    the non-Cartesian case.

    Parameters
    ----------
    kspace_loc: np.ndarray
        The sample locations previously normalized around [-0.5; 0.5[
        using Kmax.
    kspace_data: np.ndarray
        The samples corresponding to kspace_loc defined above.

    Returns
    -------
    reduced_kspace_loc: np.ndarray
        The sample locations reduced strictly to [-0.5; 0.5[ by discarding
        outliers.
    reduced_kspace_data: np.ndarray
        The samples corresponding to reduced_kspace_loc defined above.
    """
    kspace_mask = np.all((kspace_loc < 0.5) & (kspace_loc >= -0.5), axis=-1)
    kspace_loc = kspace_loc[kspace_mask]
    kspace_data = kspace_data[:, kspace_mask]
    return np.ascontiguousarray(kspace_loc), np.ascontiguousarray(kspace_data)



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
    from ..non_cartesian import NonCartesianFFT, gpuNUFFT
    if isinstance(fourier_op, NonCartesianFFT) and \
            isinstance(fourier_op.impl, gpuNUFFT):
        return fourier_op.impl.uses_sense
    else:
        return False


def estimate_density_compensation(kspace_loc, volume_shape, implementation='pipe', **kwargs):
    """ Utils function to obtain the density compensator for a
    given set of kspace locations.

    Parameters
    ----------
    kspace_loc: np.ndarray
        the kspace locations
    volume_shape: np.ndarray
        the volume shape
    implementation: str default 'pipe'
        the implementation of the non-cartesian operator
        can be 'pipe' which needs gpuNUFFT or 'cell_count'
    kwargs: dict
        extra keyword arguments to be passed to the density
        compensation estimation
    """
    density_comp = get_density(implementation)(
        kspace_loc,
        volume_shape,
        **kwargs
    )
    return density_comp
