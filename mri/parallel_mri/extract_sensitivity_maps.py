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
# System import
from mri.reconstruct.fourier import NFFT

# Package import
from scipy.interpolate import griddata
from joblib import Parallel, delayed
import scipy.fftpack as pfft

# Third party import
import numpy as np


def extract_k_space_center(samples, samples_locations,
                           thr=None, img_shape=None):
    """
    This class extract the k space center for a given threshold.

    Parameters
    ----------
    samples: np.ndarray
        The value of the samples
    samples_locations: np.ndarray
        The samples location in the k-sapec domain (between [-0.5, 0.5[)
    thr: tuple
        The threshold used to extract the k_space center
    img_shape: tuple
        The image shape to estimate the cartesian density

    Returns
    -------
    The extracted center of the k-space
    """
    if thr is None:
        if img_shape is None:
            raise ValueError('target image cartesian image shape must be fill')
        raise NotImplementedError
    else:
        samples_thresholded = np.copy(samples)
        for i in np.arange(len(thr)):
            samples_thresholded *= (samples_locations[:, i] <= thr[i])
    return samples_thresholded


def extract_k_space_center_and_locations(data_values, samples_locations,
                                         thr=None, img_shape=None):
    """
    This class extract the k space center for a given threshold and extracts
    the corresponding sampling locations

    Parameters
    ----------
    samples: np.ndarray
        The value of the samples
    samples_locations: np.ndarray
        The samples location in the k-sapec domain (between [-0.5, 0.5[)
    thr: tuple or float
        The threshold used to extract the k_space center
    img_shape: tuple
        The image shape to estimate the cartesian density

    Returns
    -------
    The extracted center of the k-space
    """
    if thr is None:
        if img_shape is None:
            raise ValueError('target image cartesian image shape must be fill')
        raise NotImplementedError
    else:
        data_thresholded = np.copy(data_values)
        condition = np.logical_and.reduce(
            tuple(np.abs(samples_locations[:, i]) <= thr[i]
                  for i in range(len(thr))))
        index = np.linspace(0, samples_locations.shape[0]-1,
                            samples_locations.shape[0], dtype=np.int)
        index = np.extract(condition, index)
        center_locations = samples_locations[index, :]
        data_thresholded = data_thresholded[:, index]
    return data_thresholded, center_locations


def grided_FT_ND(points, values, grid, method):
    """
    This function calculates the grided fourier transform
    from Interpolated non-Cartesian data into a cartesian grid

    Parameters
    ----------
    points: np.ndarray
        The N-D k_space locations of size [M, N]
    values: np.ndarray
        An image of size [N_x, N_y, N_z, ..]
    method: {'linear', 'nearest', 'cubic'}, optional
        Method of interpolation for more details see scipy.interpolate.griddata
        documentation
    grid: np.ndarray
        The Gridded matrix for which you want to calculate k_space Smaps
    Returns
    -------
    np.ndarray
        The Sensitivity map for given channel
    """
    gridded_kspace = griddata(points,
                              values,
                              grid,
                              method=method,
                              fill_value=0)
    return (np.swapaxes(pfft.fftshift(
        pfft.ifftn(pfft.ifftshift(gridded_kspace))), 1, 0))


def get_Smaps(k_space, img_shape, samples=None, mode='Gridding',
              min_samples=(-0.5, -0.5, -0.5), method='linear',
              max_samples=(0.5, 0.5, 0.5), n_cpu=1):
    """
    This method estimate the sensitivity maps information from parallel mri
    acquisition and for variable density sampling scheme where teh k-space
    center had been heavily sampled.

    Parameters
    ----------
    k_space: np.ndarray
        The acquired kspace of shape (M,L), where M is the number of samples
        acquired and L is the number of coils used
    samples: np.ndarray
    mode: string 'FFT' | 'NFFT' | 'gridding'
        Defines the mode in which we would want to interpolate
    method: string 'linear' | 'cubic' | 'nearest'
        For gridding mode, it defines the way interpolation must be done
    min_samples: tuple
        The minimum value in k-space where gridding must be done
    max_samples: tuple
        The maximum value in k-space where gridding must be done
    n_cpu: int
        Number of parallel jobs in case of parallel MRI

    Returns
    -------
    Smaps: np.ndarray
        the estimated sensitivity maps of shape (img_shape, L) with L the
        number of channels
    ISOS: np.ndarray
        The sum of Square used to extract the sensitivity maps
        :param method:
    """
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
        fourier_op = NFFT(samples=samples, shape=img_shape)
        Smaps = np.asarray([fourier_op.adj_op(k_space[l]) for l in range(L)])
    else:
        Smaps = []
        grid_space = [np.linspace(min_samples[i],
                                  max_samples[i],
                                  num=img_shape[i],
                                  endpoint=False)
                      for i in np.arange(np.size(img_shape))]
        grid = np.meshgrid(*grid_space)
        Smaps = \
            Parallel(n_jobs=n_cpu)(delayed(grided_FT_ND)
                                   (points=samples,
                                    values=k_space[l],
                                    grid=tuple(grid),
                                    method=method) for l in range(L))
        Smaps = np.asarray(Smaps)
    SOS = np.sqrt(np.sum(np.abs(Smaps)**2, axis=0))
    for r in range(L):
        Smaps[r] /= SOS
    return Smaps, SOS
