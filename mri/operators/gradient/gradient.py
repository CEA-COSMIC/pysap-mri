# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains classes for defining algorithm operators and gradients.
"""

# Package import
from .base import GradBaseMRI

# Third party import
import numpy as np


class GradAnalysis(GradBaseMRI):
    """ Gradient Analysis class.
    This class defines the grad operators for:
            (1/2) * sum(||Ft x - yl||^2_2,l)
    Attributes
    ----------
    data: np.ndarray
        input data array.
    fourier_op: instance
        a Fourier operator instance.
    verbose: int, default 0
        Debug verbosity. Prints debug information during initialization if 1.
    """
    def __init__(self, data, fourier_op, verbose=0, **kwargs):
        if fourier_op.n_coils != 1:
            super(GradAnalysis, self).__init__(data, fourier_op.op,
                                               fourier_op.adj_op,
                                               (fourier_op.n_coils,
                                                *fourier_op.shape),
                                               verbose=verbose,
                                               **kwargs)
        else:
            super(GradAnalysis, self).__init__(data, fourier_op.op,
                                               fourier_op.adj_op,
                                               fourier_op.shape,
                                               verbose=verbose,
                                               **kwargs)
        self.fourier_op = fourier_op


class GradSynthesis(GradBaseMRI):
    """ Gradient Synthesis class.
    This class defines the grad operators for:
            (1/2) * sum(||Ft Psi_t alpha - yl||^2_2,l)
    Attributes
    ----------
    data: np.ndarray
        input 2D data array.
    fourier_op: instance
        a Fourier operator instance.
    linear_op: instance
        a linear operator instance.
    verbose: int, default 0
        Debug verbosity. Prints debug information during initialization if 1.
    """
    def __init__(self, data, linear_op, fourier_op, verbose=0, **kwargs):
        self.fourier_op = fourier_op
        self.linear_op = linear_op
        coef = linear_op.op(np.squeeze(np.zeros((linear_op.n_coils,
                                                 *fourier_op.shape))))
        self.linear_op_coeffs_shape = coef.shape
        super(GradSynthesis, self).__init__(data,
                                            self._op_method,
                                            self._trans_op_method,
                                            self.linear_op_coeffs_shape,
                                            verbose=verbose,
                                            **kwargs)

    def _op_method(self, data):
        return self.fourier_op.op(self.linear_op.adj_op(data))

    def _trans_op_method(self, data):
        return self.linear_op.op(self.fourier_op.adj_op(data))


class GradSelfCalibrationAnalysis(GradBaseMRI):
    """ Gradient Analysis class for parallel MR reconstruction based on the
    coil sensitivity profile.
    This class defines the grad operators for:
            (1/2) * sum(||Ft Sl x - yl||^2_2,l)
    Attributes
    ----------
    data: np.ndarray
        input observed data array.
    fourier_op: instance
        a Fourier operator instance.
    Smaps: np.ndarray
        Coil sensitivity profile [L, Nx, Ny, Nz]
    verbose: int, default 0
        Debug verbosity. Prints debug information during initialization if 1.
    """
    def __init__(self, data, fourier_op, Smaps, verbose=0, **kwargs):
        self.Smaps = Smaps
        self.fourier_op = fourier_op
        super(GradSelfCalibrationAnalysis, self).__init__(
            data,
            self._op_method,
            self._trans_op_method,
            fourier_op.shape,
            verbose=verbose,
            **kwargs)

    def _op_method(self, data):
        data_per_ch = np.asarray([data * self.Smaps[ch]
                                  for ch in range(self.Smaps.shape[0])])
        return self.fourier_op.op(data_per_ch)

    def _trans_op_method(self, coeff):
        data_per_ch = self.fourier_op.adj_op(coeff)
        return np.sum(data_per_ch * np.conjugate(self.Smaps), axis=0)


class GradSelfCalibrationSynthesis(GradBaseMRI):
    """ Gradient Synthesis class for parallel MR reconstruction based on the
    coil sensitivity profile.
    This class defines the grad operators for:
            (1/2) * sum(||Ft Psi_t Sl Alpha - yl||^2_2,l)
    Attributes
    ----------
    data: np.ndarray
        input observed data array.
    fourier_op: instance
        a Fourier operator instance.
    linear_op: instance
        a linear operator instance.
    Smaps: np.ndarray
        Coil sensitivity profile [L, Nx, Ny, Nz]
    verbose: int, default 0
        Debug verbosity. Prints debug information during initialization if 1
    """
    def __init__(self, data, fourier_op, linear_op, Smaps, verbose=0,
                 **kwargs):
        self.Smaps = Smaps
        self.fourier_op = fourier_op
        self.linear_op = linear_op
        coef = linear_op.op(np.zeros(fourier_op.shape))
        self.linear_op_coeffs_shape = coef.shape
        super(GradSelfCalibrationSynthesis, self).__init__(
            data,
            self._op_method,
            self._trans_op_method,
            self.linear_op_coeffs_shape,
            verbose=verbose,
            **kwargs)

    def _op_method(self, coeff):
        image = self.linear_op.adj_op(coeff)
        image_per_ch = np.asarray([image * self.Smaps[ch]
                                   for ch in range(self.Smaps.shape[0])])
        return self.fourier_op.op(image_per_ch)

    def _trans_op_method(self, data):
        data_per_ch = self.fourier_op.adj_op(data)
        image_recon = np.sum(data_per_ch * np.conjugate(self.Smaps), axis=0)
        return self.linear_op.op(image_recon)
