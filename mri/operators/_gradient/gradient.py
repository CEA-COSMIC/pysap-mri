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
from ._base import GradBaseMRI

# Third party import
import numpy as np


class GradAnalysis(GradBaseMRI):
    def __init__(self, data, fourier_op, **kwargs):
        super(GradAnalysis, self).__init__(data, fourier_op.op,
                                           fourier_op.adj_op,
                                           fourier_op.shape,
                                           **kwargs)
        self.fourier_op = fourier_op


class GradSynthesis(GradBaseMRI):
    def __init__(self, data, linear_op, fourier_op, **kwargs):
        self.fourier_op = fourier_op
        self.linear_op = linear_op
        coef = linear_op.op(np.zeros(fourier_op.shape))
        self.linear_op_coeffs_shape = coef.shape
        super(GradSynthesis, self).__init__(data,
                                            self._op_method,
                                            self._trans_op_method,
                                            self.linear_op_coeffs_shape,
                                            **kwargs)

    def _op_method(self, data):
        return self.fourier_op.op(self.linear_op.adj_op(data))

    def _trans_op_method(self, data):
        return self.linear_op.op(self.fourier_op.adj_op(data))


class GradSelfCalibrationAnalysis(GradBaseMRI):
    def __init__(self, data, fourier_op, Smaps, **kwargs):
        super(GradSelfCalibrationAnalysis, self).__init__(
            data,
            self._op_method,
            self._trans_op_method,
            fourier_op.shape,
            **kwargs)
        self.Smaps = Smaps
        self.fourier_op = fourier_op

    def _op_method(self, data):
        data_per_ch = np.asarray([data * self.Smaps[ch]
                                  for ch in range(self.Smaps.shape[0])])
        return self.fourier_op.op(data_per_ch)

    def _trans_op_method(self, coeff):
        data_per_ch = self.fourier_op.adj_op(coeff)
        return np.sum(data_per_ch * np.conjugate(self.Smaps), axis=0)


class GradSelfCalibrationSynthesis(GradBaseMRI):
    def __init__(self, data, fourier_op, linear_op, Smaps, **kwargs):
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
