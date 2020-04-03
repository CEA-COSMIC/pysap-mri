# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains classes for defining gradients.
"""

# Package import
from .base import GradBaseMRI

# Third party import
import numpy as np


class GradAnalysis(GradBaseMRI):
    """ Gradient Analysis class.
    This class defines the grad operators for:
    (1/2) * sum(||F x - yl||^2_2,l)

    Attributes
    ----------
    fourier_op: an object of class in mri.operators.fourier
        a Fourier operator from FFT, NonCartesianFFT or Stacked3DNFFT
        This is F in above equation.
    verbose: int, default 0
        Debug verbosity. Prints debug information during initialization if 1.
    """
    def __init__(self, fourier_op, verbose=0, **kwargs):
        if fourier_op.n_coils != 1:
            data_shape = (fourier_op.n_coils, *fourier_op.shape)
        else:
            data_shape = fourier_op.shape
        super(GradAnalysis, self).__init__(
            operator=fourier_op.op,
            trans_operator=fourier_op.adj_op,
            shape=data_shape,
            verbose=verbose,
            **kwargs,
        )
        self.fourier_op = fourier_op


class GradSynthesis(GradBaseMRI):
    """ Gradient Synthesis class.
    This class defines the grad operators for:
    (1/2) * sum(||F Psi_t alpha - yl||^2_2,l)

    Attributes
    ----------
    fourier_op: an object of class in mri.operators.fourier
        a Fourier operator from FFT, NonCartesianFFT or Stacked3DNFFT
        This is F in above equation.
    linear_op: an object of class in mri.operators.linear
        a linear operator from WaveltN or WaveletUD2
        This is Psi in above equation.
    verbose: int, default 0
        Debug verbosity. Prints debug information during initialization if 1.
    """
    def __init__(self, linear_op, fourier_op, verbose=0, **kwargs):
        self.fourier_op = fourier_op
        self.linear_op = linear_op
        coef = linear_op.op(np.squeeze(np.zeros((linear_op.n_coils,
                                                 *fourier_op.shape))))
        self.linear_op_coeffs_shape = coef.shape
        super(GradSynthesis, self).__init__(
            self._op_method,
            self._trans_op_method,
            self.linear_op_coeffs_shape,
            verbose=verbose,
            **kwargs,
        )

    def _op_method(self, data):
        return self.fourier_op.op(self.linear_op.adj_op(data))

    def _trans_op_method(self, data):
        return self.linear_op.op(self.fourier_op.adj_op(data))


class GradSelfCalibrationAnalysis(GradBaseMRI):
    """ Gradient Analysis class for parallel MR reconstruction based on the
    coil sensitivity profile.
    This class defines the grad operators for:
    (1/2) * sum(||F Sl x - yl||^2_2,l)

    Attributes
    ----------
    fourier_op: an object of class in mri.operators.fourier
        a Fourier operator from FFT, NonCartesianFFT or Stacked3DNFFT
        This is F in above equation.
    Smaps: np.ndarray
        Coil sensitivity profile [L, * data.shape]
        This is collection of Sl in above equation.
    verbose: int, default 0
        Debug verbosity. Prints debug information during initialization if 1.
    """
    def __init__(self, fourier_op, Smaps, verbose=0, **kwargs):
        self.Smaps = Smaps
        self.fourier_op = fourier_op
        super(GradSelfCalibrationAnalysis, self).__init__(
            self._op_method,
            self._trans_op_method,
            fourier_op.shape,
            verbose=verbose,
            **kwargs,
        )

    def _op_method(self, data):
        data_per_ch = data * self.Smaps
        return self.fourier_op.op(data_per_ch)

    def _trans_op_method(self, coeff):
        data_per_ch = self.fourier_op.adj_op(coeff)
        return np.sum(data_per_ch * np.conjugate(self.Smaps), axis=0)


class GradSelfCalibrationSynthesis(GradBaseMRI):
    """ Gradient Synthesis class for parallel MR reconstruction based on the
    coil sensitivity profile.
    This class defines the grad operators for:
    (1/2) * sum(||F Sl Psi_t Alpha - yl||^2_2,l)

    Attributes
    ----------
    fourier_op: an object of class in mri.operators.fourier
        a Fourier operator from FFT, NonCartesianFFT or Stacked3DNFFT
        This is F in above equation.
    linear_op: an object of class in mri.operators.linear
        a linear operator from WaveletN or WaveletUD2
        This is Psi in above equation.
    Smaps: np.ndarray
        Coil sensitivity profile [L, * data.shape]
        This is collection of Sl in above equation.
    verbose: int, default 0
        Debug verbosity. Prints debug information during initialization if 1
    """
    def __init__(self, fourier_op, linear_op, Smaps, verbose=0,
                 **kwargs):
        self.Smaps = Smaps
        self.fourier_op = fourier_op
        self.linear_op = linear_op
        coef = linear_op.op(np.zeros(fourier_op.shape))
        self.linear_op_coeffs_shape = coef.shape
        super(GradSelfCalibrationSynthesis, self).__init__(
            self._op_method,
            self._trans_op_method,
            self.linear_op_coeffs_shape,
            verbose=verbose,
            **kwargs,
        )

    def _op_method(self, coeff):
        image = self.linear_op.adj_op(coeff)
        image_per_ch = image * self.Smaps
        return self.fourier_op.op(image_per_ch)

    def _trans_op_method(self, data):
        data_per_ch = self.fourier_op.adj_op(data)
        image_recon = np.sum(data_per_ch * np.conjugate(self.Smaps), axis=0)
        return self.linear_op.op(image_recon)
