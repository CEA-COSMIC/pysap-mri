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
from copy import deepcopy


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
    thr: float
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
        for i in np.arange(np.size(thr)):
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
        center_locations = np.copy(samples_locations)
        condn = [np.abs(samples_locations[:, i]) <= thr[i]
                 for i in np.arange(np.size(thr))]
        condition = np.ones(data_values.shape[1], dtype=bool)
        for i in np.arange(len(condn)):
            condition = np.logical_and(condn[i], condition)
        index = np.linspace(0, samples_locations.shape[0]-1,
                            samples_locations.shape[0], dtype=np.int)
        index = np.extract(condition, index)
        center_locations = samples_locations[index, :]
        data_thresholded = data_thresholded[:, index]
    return data_thresholded, center_locations


def gridding_nd(points, values, img_shape, grid, method='linear'):
    """
    Interpolate non-Cartesian data into a cartesian grid

    Parameters
    ----------
    points: np.ndarray
        The 2D k_space locations of size [M, 2]
    values: np.ndarray
        An image of size [N_x, N_y]
    img_shape: tuple
        The final output ndarray
    method: {'linear', 'nearest', 'cubic'}, optional
        Method of interpolation for more details see scipy.interpolate.griddata
        documentation
    points_min: float
        The minimum points in the gridded matrix, if not provide take the min
        of points
    points_max: float
        The maximum points in the gridded matrix, if not provide take the min
        of points
    Returns
    -------
    np.ndarray
        The gridded solution of shape [N_x, N_y]
    """
    return griddata(points,
                    values,
                    grid,
                    method=method,
                    fill_value=0)


def get_Smaps(k_space, img_shape, samples=None, mode='Gridding',
              min_samples=None, max_samples=None, n_cpu=1):
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

    Returns
    -------
    Smaps: np.ndarray
        the estimated sensitivity maps of shape (img_shape, L) with L the
        number of channels
    ISOS: np.ndarray
        The sum of Square used to extract the sensitivity maps
    """
    if min_samples is None:
        min_samples = np.min(samples, axis=0)
    if max_samples is None:
        max_samples = np.max(samples, axis=0)

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
        gridded_kspaces = \
            Parallel(n_jobs=n_cpu)(delayed(gridding_nd)
                                   (points=np.copy(samples),
                                    values=np.copy(k_space[l]),
                                    img_shape=img_shape,
                                    grid=tuple(grid),
                                    method='linear') for l in range(L))
        for gridded_kspace in gridded_kspaces:
            Smaps.append(np.swapaxes(pfft.fftshift(
                pfft.ifftn(pfft.ifftshift(gridded_kspace))), 1, 0))
        Smaps = np.asarray(Smaps)
    SOS = np.sqrt(np.sum(np.abs(Smaps)**2, axis=0))
    for r in range(L):
        Smaps[r] /= SOS
    return Smaps, SOS

