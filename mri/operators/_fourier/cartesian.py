# -*- coding: utf-8 -*-
# #############################################################################
#  pySAP - Copyright (C) CEA, 2017 - 2018                                     #
#  Distributed under the terms of the CeCILL-B license,                       #
#  as published by the CEA-CNRS-INRIA. Refer to the LICENSE file or to        #
#  http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.   #
# #############################################################################

"""
Fourier operators for cartesian space.
"""

# System import
import warnings
import numpy as np

# Package import
from .._base import OperatorBase
from .utils import convert_locations_to_mask


class FFT(OperatorBase):
    """ Standard unitary ND Fast Fourrier Transform class.
    The FFT will be normalized in a symmetric way

    Attributes
    ----------
    samples: np.ndarray
        the mask samples in the Fourier domain.
    shape: tuple of int
        shape of the image (not necessarly a square matrix).
     n_coils: int, default 1
            Number of coils used to acquire the signal in case of multiarray
            receiver coils acquisition. If n_coils > 1, data shape must be
            [n_coils, Nx, Ny, NZ]
    """

    def __init__(self, samples, shape, n_coils=1):
        self.samples = samples
        self.shape = shape
        self._mask = convert_locations_to_mask(self.samples, self.shape)
        if n_coils <= 0:
            warnings.warn("The number of coils should be strictly positive")
            n_coils = 1
        self.n_coils = n_coils

    def op(self, img):
        """ This method calculates the masked Fourier transform of a ND image.

        Parameters
        ----------
        img: np.ndarray
            input ND array with the same shape as the mask. For multichannel
            images the coils dimension is put first

        Returns
        -------
        x: np.ndarray
            masked Fourier transform of the input image. For multichannel
            images the coils dimension is put first
        """
        if self.n_coils == 1:
            return self._mask * np.fft.ifftshift(np.fft.fftn(
                np.fft.fftshift(img), norm="ortho"))
        else:
            if self.n_coils > 1 and self.n_coils != img.shape[0]:
                raise ValueError("The number of coils parameter is not equal"
                                 "to the actual number of coils, the data must"
                                 "be reshaped as [n_coils, Nx, Ny, Nz]")
            else:
                # TODO: Use joblib for parallelization
                return np.asarray([self._mask * np.fft.ifftshift(np.fft.fftn(
                    np.fft.fftshift(img[ch]), norm="ortho"))
                    for ch in range(self.n_coils)])

    def adj_op(self, x):
        """ This method calculates inverse masked Fourier transform of a ND
        image.

        Parameters
        ----------
        x: np.ndarray
            masked Fourier transform data. For multichannel
            images the coils dimension is put first

        Returns
        -------
        img: np.ndarray
            inverse ND discrete Fourier transform of the input coefficients.
            For multichannel images the coils dimension is put first
        """
        if self.n_coils == 1:
            return np.fft.fftshift(np.fft.ifftn(
                np.fft.ifftshift(self._mask * x), norm="ortho"))
        else:
            if self.n_coils > 1 and self.n_coils != x.shape[0]:
                raise ValueError("The number of coils parameter is not equal"
                                 "to the actual number of coils, the data must"
                                 "be reshaped as [n_coils, Nx, Ny, Nz]")
            else:
                # TODO: Use joblib for parallelization
                return np.asarray([np.fft.fftshift(np.fft.ifftn(
                    np.fft.ifftshift(self._mask * x[ch]),
                    norm="ortho"))
                    for ch in range(self.n_coils)])
