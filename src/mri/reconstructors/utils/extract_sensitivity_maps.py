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
from mri.operators.utils import gridded_inverse_fourier_transform_nd, \
    convert_locations_to_mask

# Third party import
from joblib import Parallel, delayed
import numpy as np
import scipy.fftpack as pfft


def extract_k_space_center_and_locations(data_values, samples_locations,
                                         thr=None, img_shape=None, window_fun=None,
                                         is_fft=False, density_comp=None):
    r"""
    This class extract the k space center for a given threshold and extracts
    the corresponding sampling locations

    Parameters
    ----------
    data_values: numpy.ndarray
        The value of the samples
    samples_locations: numpy.ndarray
        The samples location in the k-sapec domain (between [-0.5, 0.5[)
    thr: tuple or float
        The threshold used to extract the k_space center
    img_shape: tuple
        The image shape to estimate the cartesian density
    is_fft: bool default False
        Checks if the incoming data is from FFT, in which case, masking
        can be done more directly
    density_comp: numpy.ndarray default None
        The density compensation for kspace data in case it exists and we
        use density compensated adjoint for Smap estimation

    window_fun: "Hann", "Hanning", "Hamming", or a callable, default None.
        The window function to apply to the selected data. It is computed with
        the center locations selected. Only works with circular mask.
        If window_fun is a callable, it takes as input the array (n_samples x n_dims)
        of sample positions and returns an array of n_samples weights to be
        applied to the selected k-space values, before the smaps estimation.


    Returns
    -------
    The extracted center of the k-space, i.e. both the kspace locations and
    kspace values. If the density compensators are passed, the corresponding
    compensators for the center of k-space data will also be returned. The
    return stypes for density compensation and kspace data is same as input

    Notes
    -----

    The Hann (or Hanning) and Hamming windows  of width :math:`2\theta` are defined as:
    .. math::

    w(x,y) = a_0 - (1-a_0) * \cos(\pi * \sqrt{x^2+y^2}/\theta),
    \sqrt{x^2+y^2} \le \theta

    In the case of Hann window :math:`a_0=0.5`.
    For Hamming window we consider the optimal value in the equiripple sense:
    :math:`a_0=0.53836`.
    .. Wikipedia:: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows

    """
    if thr is None:
        if img_shape is None:
            raise ValueError('target image cartesian image shape must be fill')
        raise NotImplementedError
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
    if window_fun is None:
        if isinstance(thr, float):
            thr = (thr,) * samples_locations.shape[1]

        condition = np.logical_and.reduce(
            tuple(np.abs(samples_locations[:, i]) <= thr[i]
                  for i in range(len(thr))))
    elif isinstance(thr, float):
        condition = np.sum(np.square(samples_locations), axis=1) <= thr**2
    else:
        raise ValueError("threshold type is not supported with select window")
    index = np.linspace(0, samples_locations.shape[0] - 1,
                        samples_locations.shape[0], dtype=np.int64)
    index = np.extract(condition, index)
    center_locations = samples_locations[index, :]
    data_thresholded = data_ordered[:, index]
    if window_fun is not None:
        if callable(window_fun):
            window = window_fun(center_locations)
        else:
            if window_fun == "Hann" or window_fun == "Hanning":
                a_0 = 0.5
            elif window_fun == "Hamming":
                a_0 = 0.53836
            else:
                raise ValueError("Unsupported window function.")

            radius = np.linalg.norm(center_locations, axis=1)
            window = a_0 + (1 - a_0) * np.cos(np.pi * radius / thr)
        data_thresholded = window * data_thresholded

    if density_comp is not None:
        density_comp = density_comp[index]
        return data_thresholded, center_locations, density_comp
    else:
        return data_thresholded, center_locations


def get_Smaps(k_space, img_shape, samples, thresh,
              min_samples, max_samples, mode='NFFT',
              method='linear',
              window_fun=None,
              density_comp=None, n_cpu=1,
              fourier_op_kwargs={}):
    r"""
    Get Smaps for from pMRI sample data.

    Estimate the sensitivity maps information from parallel mri
    acquisition and for variable density sampling scheme where the k-space
    center had been heavily sampled.

    Reference : Self-Calibrating Nonlinear Reconstruction Algorithms for
    Variable Density Sampling and Parallel Reception MRI
    https://ieeexplore.ieee.org/abstract/document/8448776

    Parameters
    ----------
    k_space: numpy.ndarray
        The acquired kspace of shape (M,L), where M is the number of samples
        acquired and L is the number of coils used
    img_shape: tuple
        The final output shape of Sensitivity Maps.
    samples: numpy.ndarray
        The sample locations where the above k_space data was acquired
    thresh: tuple
        The value of threshold in kspace for thresholding k-space center
    min_samples: tuple
        The minimum values in k-space where gridding must be done
    max_samples: tuple
        The maximum values in k-space where gridding must be done
    mode: string 'FFT' | 'NFFT' | 'gridding' , default='NFFT'
        Defines the mode in which we would want to interpolate,
        NOTE: FFT should be considered only if the input has
        been sampled on the grid
    method: string 'linear' | 'cubic' | 'nearest', default='linear'
        For gridding mode, it defines the way interpolation must be done
    window_fun: "Hann", "Hanning", "Hamming", or a callable, default None.
        The window function to apply to the selected data. It is computed with
        the center locations selected. Only works with circular mask.
        If window_fun is a callable, it takes as input the n_samples x n_dims
        of samples position and would return an array of n_samples weight to be
        applied to the selected k-space values, before the smaps estimation.
    density_comp: numpy.ndarray default None
        The density compensation for kspace data in case it exists and we
        use density compensated adjoint for Smap estimation
    n_cpu: int default=1
        Number of parallel jobs in case of parallel MRI
    fourier_op_kwargs: dict, default {}
        The keyword arguments given to fourier_op initialization if
        mode == 'NFFT'. If None, we choose implementation of fourier op to
        'gpuNUFFT' if library is installed.

    Returns
    -------
    Smaps: numpy.ndarray
        the estimated sensitivity maps of shape (img_shape, L) with L the
        number of channels
    SOS: numpy.ndarray
        The sum of Square used to extract the sensitivity maps

    Notes
    -----

    The Hann (or Hanning) and Hamming window  of width :math:`2\theta` are defined as:
    .. math::

    w(x,y) = a_0 - (1-a_0) * \cos(\pi * \sqrt{x^2+y^2}/\theta),
    \sqrt{x^2+y^2} \le \theta

    In the case of Hann window :math:`a_0=0.5`.
    For Hamming window we consider the optimal value in the equiripple sens:
    :math:`a_0=0.53836`.
    .. Wikipedia:: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows


    """
    if len(min_samples) != len(img_shape) \
            or len(max_samples) != len(img_shape):
        raise NameError('The img_shape, max_samples, and '
                        'min_samples must be of same length')
    k_space, samples, *density_comp = \
        extract_k_space_center_and_locations(
            data_values=k_space,
            samples_locations=samples,
            thr=thresh,
            img_shape=img_shape,
            is_fft=mode == 'FFT',
            window_fun=window_fun,
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
        fourier_op = NonCartesianFFT(
            samples=samples,
            shape=img_shape,
            density_comp=density_comp,
            n_coils=L,
            **fourier_op_kwargs,
        )
        Smaps = fourier_op.adj_op(np.ascontiguousarray(k_space))
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
