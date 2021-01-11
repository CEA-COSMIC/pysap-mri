# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Fourier operators for cartesian and non-cartesian space.
"""

# System import
import warnings
import numpy as np
import scipy as sp

# Package import
from ..base import OperatorBase
from .utils import convert_locations_to_mask
from modopt.interface.errors import warn

# Third party import
try:
    import pynfft
except Exception:
    warnings.warn("pynfft python package has not been found. If needed use "
                  "the master release.")
    pass
try:
    from pynufft import NUFFT_hsa, NUFFT_cpu
except Exception:
    warnings.warn("pynufft python package has not been found. If needed use "
                  "the master release. Till then you cannot use NUFFT on GPU")
    pass


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
    def __init__(self, shape, n_coils=1, samples=None, mask=None):
        """ Initilize the 'FFT' class.

        Parameters
        ----------
        shape: tuple of int
            shape of the image (not necessarly a square matrix).
        n_coils: int, default 1
            Number of coils used to acquire the signal in case of
            multiarray receiver coils acquisition. If n_coils > 1,
            data shape must be equal to [n_coils, Nx, Ny, NZ]
        samples: np.ndarray, default None
            the mask samples in the Fourier domain.
        mask: np.ndarray, default None
            the mask as a matrix with 1 at sample locations
            please pass samples or mask
        """
        self.shape = shape
        if mask is None and samples is None:
            raise ValueError("Please pass either samples or mask as input")
        if mask is None:
            self.mask = convert_locations_to_mask(samples, self.shape)
        else:
            self.mask = mask
        if n_coils <= 0:
            warn("The number of coils should be strictly positive")
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
            return self.mask * sp.fft.ifftshift(sp.fft.fftn(
                                    sp.fft.fftshift(img), norm="ortho"))
        else:
            if self.n_coils > 1 and self.n_coils != img.shape[0]:
                raise ValueError("The number of coils parameter is not equal"
                                 "to the actual number of coils, the data must"
                                 "be reshaped as [n_coils, Nx, Ny, Nz]")
            else:
                axes = tuple(np.arange(1, img.ndim))
                return self.mask * sp.fft.ifftshift(
                    sp.fft.fftn(
                        sp.fft.fftshift(
                            img,
                            axes=axes
                        ),
                        axes=axes,
                        norm="ortho",
                    ),
                    axes=axes
                )

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
            return sp.fft.fftshift(sp.fft.ifftn(
                        sp.fft.ifftshift(self.mask * x), norm="ortho"))
        else:
            if self.n_coils > 1 and self.n_coils != x.shape[0]:
                raise ValueError("The number of coils parameter is not equal"
                                 "to the actual number of coils, the data must"
                                 "be reshaped as [n_coils, Nx, Ny, Nz]")
            else:
                x = x * self.mask
                axes = tuple(np.arange(1, x.ndim))
                return sp.fft.fftshift(
                    sp.fft.ifftn(
                        sp.fft.ifftshift(
                            x,
                            axes=axes
                        ),
                        axes=axes,
                        norm="ortho",
                    ),
                    axes=axes
                )
