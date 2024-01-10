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

# Package import
from ..base import OperatorBase
from mrinufft import get_operator


class NonCartesianFFT(OperatorBase):
    """This class wraps around different implementation algorithms for NFFT"""
    def __init__(self, samples, shape, implementation='finufft', n_coils=1,
                 density_comp=None, **kwargs):
        """ A small wrapper around mri-nufft package
        This is mostly maintained just for legacy reasons (all legacy reconstruction
        uses this codes)

        Parameters
        ----------
        samples: np.ndarray (Mxd)
            the samples locations in the Fourier domain where M is the number
            of samples and d is the dimensionality of the output data
            (2D for an image, 3D for a volume).
        shape: tuple of int
            shape of the image (not necessarily a square matrix).
        implementation: str 'finufft' | 'cufinufft' | 'gpuNUFFT',
        default 'finufft'
            which implementation of NFFT to use.
        n_coils: int default 1
            Number of coils used to acquire the signal in case of multiarray
            receiver coils acquisition
        density_comp: np.ndarray (M,) default None
        kwargs: extra keyword args
            these arguments are passed to underlying operator. Check the docs of 
            mrinufft : https://mind-inria.github.io/mri-nufft/
        """
        self.shape = shape
        self.samples = samples
        self.n_coils = n_coils
        self.implementation = implementation
        self.density_comp = density_comp
        self.kwargs = kwargs
        self.impl = get_operator(self.implementation)(
            self.samples,
            self.shape,
            density=self.density_comp,
            n_coils=self.n_coils,
            **self.kwargs
        )
        
        

    def op(self, data, *args):
        """ This method calculates the masked non-cartesian Fourier transform
        of an image.

        Parameters
        ----------
        img: np.ndarray
            input N-D array with the same shape as shape.

        Returns
        -------
            masked Fourier transform of the input image.
        """
        return self.impl.op(data, *args)

    def adj_op(self, coeffs, *args):
        """ This method calculates inverse masked non-uniform Fourier
        transform of a 1-D coefficients array.

        Parameters
        ----------
        x: np.ndarray
            masked non-uniform Fourier transform 1D data.

        Returns
        -------
            inverse discrete Fourier transform of the input coefficients.
        """
        return self.impl.adj_op(coeffs, *args)

