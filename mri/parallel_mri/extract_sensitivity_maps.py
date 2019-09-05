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


def gridding_3d(points, values, xi, method='linear', fill_value=0):
    """
    This method is perform the gridding operations.It is based on the griddata
    function from scipy.interpolate.
    Parameters:
    ----------
    points: np.ndarray
        The kspace frequency locations of shape [M, d], where M is the total
        number of points and d is the dimension of the k-space
    value: np.ndarray
        The acquired kspace of shape (M,L), where M is the number of samples
        acquired and L is the number of coils used
    xi: 2-D ndarray of float or tuple of 1-D array, shape (N, D)
        Points at which to interpolate data
    method : {'linear', 'nearest', 'cubic'}, optional
        Method of interpolation. One of
        ``nearest`` return the value at the data point closest to
          the point of interpolation.
        ``linear`` tesselate the input point set to n-dimensional simplices,
          and interpolate linearly on each simplex.
        ``cubic`` (1-D) return the value determined from a cubic spline.
        ``cubic`` (2-D) return the value determined from a
          piecewise cubic, continuously differentiable (C1), and
          approximately curvature-minimizing polynomial surface.
    fill_value : float, optional
        Value used to fill in for requested points outside of the
        convex hull of the input points.  If not provided, then the
        default is ``nan``. This option has no effect for the
        'nearest' method.
    Returns:
    -------
    The gridded value
    """
    return griddata(points=np.copy(points),
                    values=np.copy(values),
                    xi=deepcopy(xi),
                    method=method,
                    fill_value=fill_value)


def get_3D_smaps(k_space, img_shape, samples=None, mode='gridding',
                 samples_min=None, samples_max=None, n_cpu=1):
    """
    This method estimate the sensitivity maps information from parallel mri
    acquisition and for variable density sampling scheme where the k-space
    center had been heavily sampled in a 3D setting.
    Parameters:
    ----------
    k_space: np.ndarray
        The acquired kspace of shape (M,L), where M is the number of samples
        acquired and L is the number of coils used
    img_shape: a 3 element tuple
        target image shape
    mode: string
        The extraction mode either:
        ``gridding`` uses the gridding operations from scipy.interpolate to
        project the k-space into a cartesian grid before doing the Fourier
        operations. This mode requires the samples
        ``FFT`` The k-space is cartesian and it only computes the SOS and
        extract Smaps by computing the ratio between the image coil and the SOS
        ``NFFT`` The NFFT is used to compute the image coil
    samples: np.ndarray
        the non-cartesian samples locations in the k-space domain could be all
        the non-cartesian samples or just a part of the frequency locations
    samples_min: np.ndarray
        The minimum samples value of the entire k-space typically -0.5
        if any is provided the min(samples) will be taken
    samples_max:
        The maximum samples value of the entire k-space typically <0.5
        if any is provided the max(samples) will be taken
    n_cpu:
        The number of cpu used to compute the gridding operations, it will
        split across the number of channels
    Returns:
    -------
    Smaps: np.ndarray
        the estimated sensitivity maps of shape (img_shape, L) with L the
        number of channels
    ISOS: np.ndarray
        The sum of Squarre used to extract the sensitivity maps
    """
    if samples_min is None:
        samples_min = [np.min(samples[:, idx]) for idx in
                       range(samples.shape[1])]
    if samples_max is None:
        samples_max = [np.max(samples[:, idx]) for idx in
                       range(samples.shape[1])]

    if samples is None:
        mode = 'FFT'

    L, M = k_space.shape
    Smaps = []
    if mode == 'FFT':
        if not M == img_shape[0]*img_shape[1]*img_shape[2]:
            raise ValueError(['The number of samples in the k-space must be',
                              'equal to the (image size, the number of coils)'
                              ])
        k_space = k_space.reshape(L, *img_shape)
        for l in range(L):
            Smaps.append(pfft.fftshift(pfft.ifftn(pfft.ifftshift(k_space[l]))))
    elif mode == 'NFFT':
        raise ValueError('NotImplemented yet')
    else:
        xi = np.linspace(samples_min[0],
                         samples_max[0],
                         num=img_shape[0],
                         endpoint=False)
        yi = np.linspace(samples_min[1],
                         samples_max[1],
                         num=img_shape[1],
                         endpoint=False)
        zi = np.linspace(samples_min[2],
                         samples_max[2],
                         num=img_shape[2],
                         endpoint=False)
        gridx, gridy, gridz = np.meshgrid(xi, yi, zi)

        gridded_kspaces = Parallel(n_jobs=n_cpu,
                                   verbose=1000)(
            delayed(gridding_3d)
            (points=np.copy(samples),
             values=np.copy(k_space[l]),
             xi=(gridx, gridy, gridz),
             method='linear',
             fill_value=0) for l in range(L))

        for gridded_kspace in gridded_kspaces:
            Smaps.append(np.swapaxes(pfft.fftshift(
                pfft.ifftn(pfft.ifftshift(gridded_kspace))), 1, 0))

    Smaps = np.asarray(Smaps)
    SOS = np.squeeze(np.sqrt(np.sum(np.abs(Smaps)**2, axis=0)))
    for l in range(L):
        Smaps[l] /= SOS
    return Smaps, SOS
