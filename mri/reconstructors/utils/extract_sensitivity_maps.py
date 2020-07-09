# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains tools to extract sensitivity maps from undersampled MR
acquisition with high density in the k space center.
"""
import warnings

# Package import
from mri.operators import NonCartesianFFT
from mri.operators.fourier.non_cartesian import gpunufft_available
from mri.operators.utils import get_stacks_fourier, \
    gridded_inverse_fourier_transform_nd, \
    gridded_inverse_fourier_transform_stack, convert_locations_to_mask

# Third party import
from joblib import Parallel, delayed
import numpy as np
import scipy.fftpack as pfft


def extract_k_space_center_and_locations(data_values, samples_locations,
                                         thr=None, img_shape=None,
                                         is_fft=False, density_comp=None):
    """
    This class extract the k space center for a given threshold and extracts
    the corresponding sampling locations

    Parameters
    ----------
    data_values: np.ndarray
        The value of the samples
    samples_locations: np.ndarray
        The samples location in the k-sapec domain (between [-0.5, 0.5[)
    thr: tuple or float
        The threshold used to extract the k_space center
    img_shape: tuple
        The image shape to estimate the cartesian density
    is_fft: bool default False
        Checks if the incoming data is from FFT, in which case, masking
        can be done more directly
    density_comp: np.ndarray default None
        The density compensation for kspace data in case it exists and we
        use density compensated adjoint for Smap estimation
    Returns
    -------
    The extracted center of the k-space, i.e. both the kspace locations and
    kspace values. If the density compensators are passed, the corresponding
    compensators for the center of k-space data will also be returned. The
    return stypes for density compensation and kspace data is same as input
    """
    if thr is None:
        if img_shape is None:
            raise ValueError('target image cartesian image shape must be fill')
        raise NotImplementedError
    else:
        if data_values.ndim > 2:
            warnings.warn('Data Values seem to have rank '
                          + str(data_values.ndim) +
                          ' (>2). Using is_fft for now.')
            is_fft = True
        if is_fft:
            img_shape = np.asarray(data_values[0].shape)
            mask = convert_locations_to_mask(samples_locations, img_shape)
            indices = np.where(np.reshape(mask, mask.size))[0]
            data_ordered = np.asarray([
                np.reshape(data_values[channel], mask.size)[indices]
                for channel in range(data_values.shape[0])])
        else:
            data_ordered = np.copy(data_values)
        condition = np.logical_and.reduce(
            tuple(np.abs(samples_locations[:, i]) <= thr[i]
                  for i in range(len(thr))))
        index = np.linspace(0, samples_locations.shape[0]-1,
                            samples_locations.shape[0], dtype=np.int)
        index = np.extract(condition, index)
        center_locations = samples_locations[index, :]
        data_thresholded = data_ordered[:, index]
        if density_comp is not None:
            density_comp = density_comp[index]
            return data_thresholded, center_locations, density_comp
        else:
            return data_thresholded, center_locations


def get_Smaps(k_space, img_shape, samples, thresh,
              min_samples, max_samples, mode='Gridding',
              method='linear', density_comp=None, n_cpu=1,
              fourier_op_kwargs=None):
    """
    This method estimate the sensitivity maps information from parallel mri
    acquisition and for variable density sampling scheme where teh k-space
    center had been heavily sampled.
    Reference : Self-Calibrating Nonlinear Reconstruction Algorithms for
    Variable Density Sampling and Parallel Reception MRI
    https://ieeexplore.ieee.org/abstract/document/8448776

    Parameters
    ----------
    k_space: np.ndarray
        The acquired kspace of shape (M,L), where M is the number of samples
        acquired and L is the number of coils used
    img_shape: tuple
        The final output shape of Sensitivity Maps.
    samples: np.ndarray
        The sample locations where the above k_space data was acquired
    thresh: tuple
        The value of threshold in kspace for thresholding k-space center
    min_samples: tuple
        The minimum values in k-space where gridding must be done
    max_samples: tuple
        The maximum values in k-space where gridding must be done
    mode: string 'FFT' | 'NFFT' | 'gridding' | 'Stack', default='gridding'
        Defines the mode in which we would want to interpolate,
        NOTE: FFT should be considered only if the input has
        been sampled on the grid
    method: string 'linear' | 'cubic' | 'nearest', default='linear'
        For gridding mode, it defines the way interpolation must be done
    density_comp: np.ndarray default None
        The density compensation for kspace data in case it exists and we
        use density compensated adjoint for Smap estimation
    n_cpu: int default=1
        Number of parallel jobs in case of parallel MRI
    fourier_op_kwargs: dict, default None
        The keyword arguments given to fourier_op initialization if
        mode == 'NFFT'. If None, we choose implementation of fourier op to
        'gpuNUFFT' if library is installed.

    Returns
    -------
    Smaps: np.ndarray
        the estimated sensitivity maps of shape (img_shape, L) with L the
        number of channels
    ISOS: np.ndarray
        The sum of Square used to extract the sensitivity maps
    """
    if len(min_samples) != len(img_shape) \
            or len(max_samples) != len(img_shape) \
            or len(thresh) != len(img_shape):
        raise NameError('The img_shape, max_samples, '
                        'min_samples and thresh must be of same length')
    k_space, samples, *density_comp = \
        extract_k_space_center_and_locations(
            data_values=k_space,
            samples_locations=samples,
            thr=thresh,
            img_shape=img_shape,
            is_fft=mode == 'FFT',
            density_comp=density_comp,
        )
    if density_comp:
        density_comp = density_comp[0]
    else:
        density_comp = None
    if samples is None:
        mode = 'FFT'
    L, M = k_space.shape
    Smaps_shape = (L, *img_shape)
    Smaps = np.zeros(Smaps_shape).astype('complex128')
    if mode == 'FFT':
        if not M == img_shape[0]*img_shape[1]:
            raise ValueError(['The number of samples in the k-space must be',
                              'equal to the (image size, the number of coils)'
                              ])
        k_space = k_space.reshape(Smaps_shape)
        for l in range(Smaps_shape[2]):
            Smaps[l] = pfft.ifftshift(pfft.ifft2(pfft.fftshift(k_space[l])))
    elif mode == 'NFFT':
        if fourier_op_kwargs is None:
            if gpunufft_available:
                fourier_op_kwargs = {'implementation': 'gpuNUFFT'}
            else:
                fourier_op_kwargs = {}
        fourier_op = NonCartesianFFT(
            samples=samples,
            shape=img_shape,
            density_comp=density_comp,
            n_coils=L,
            **fourier_op_kwargs,
        )
        Smaps = fourier_op.adj_op(np.ascontiguousarray(k_space))
    elif mode == 'Stack':
        grid_space = [np.linspace(min_samples[i],
                                  max_samples[i],
                                  num=img_shape[i],
                                  endpoint=False)
                      for i in np.arange(np.size(img_shape)-1)]
        grid = np.meshgrid(*grid_space)
        kspace_plane_loc, _, sort_pos, idx_mask_z = \
            get_stacks_fourier(samples, img_shape)
        Smaps = Parallel(n_jobs=n_cpu, mmap_mode='r+')(
            delayed(gridded_inverse_fourier_transform_stack)(
                kspace_data_sorted=k_space[l, sort_pos],
                kspace_plane_loc=kspace_plane_loc,
                idx_mask_z=idx_mask_z,
                grid=tuple(grid),
                volume_shape=img_shape,
                method=method
            )
            for l in range(L)
        )
        Smaps = np.asarray(Smaps)
    elif mode == 'gridding':
        grid_space = [np.linspace(min_samples[i],
                                  max_samples[i],
                                  num=img_shape[i],
                                  endpoint=False)
                      for i in np.arange(np.size(img_shape))]
        grid = np.meshgrid(*grid_space)
        Smaps = \
            Parallel(n_jobs=n_cpu, verbose=1, mmap_mode='r+')(
                delayed(gridded_inverse_fourier_transform_nd)(
                    kspace_loc=samples,
                    kspace_data=k_space[l],
                    grid=tuple(grid),
                    method=method
                )
                for l in range(L)
            )
        Smaps = np.asarray(Smaps)
    else:
        raise ValueError('Bad smap_extract_mode chosen! '
                         'Please choose between : '
                         '`FFT` | `NFFT` | `gridding` | `Stack`')
    SOS = np.sqrt(np.sum(np.abs(Smaps)**2, axis=0))
    for r in range(L):
        Smaps[r] /= SOS
    return Smaps, SOS
