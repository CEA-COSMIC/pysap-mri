# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This modules contains classes for Off-Resonance Correction (ORC)

:Author: Guillaume Daval-Frérot <guillaume.davalfrerot@cea.fr>
"""

import numpy as np

from ..base import OperatorBase
from .utils import compute_orc_coefficients


class ORCFFTWrapper(OperatorBase):
    """ Off-Resonance Correction FFT Wrapper

    This class is used to wrap any Fourier operator and change it
    into an off-resonance correction multi-linear operator using
    the method described in :cite: `sutton2003` with different
    choices of coefficients described in :cite: `fessler2005`.

    """
    def __init__(self, fourier_op, field_map, time_vec, mask,
                 coefficients="svd", weights="full", num_interpolators="auto",
                 n_bins="auto"):
        """ Initialize and compute multi-linear correction coefficients

        Parameters
        ----------
        fourier_op: OperatorBase
            Cartesian or non-Cartesian Fourier operator to wrap
        field_map: numpy.ndarray
            B0 field inhomogeneity map (in Hz)
        time_vec: numpy.ndarray
            1D vector indicating time after pulse in each shot (in s)
        mask: numpy.ndarray
            Mask describing the regions to consider during correction
        coefficients: {'svd', 'mti', 'mfi'}, optional
            Type of interpolation coefficients to use (default is 'svd')
        weights: {'full', 'sqrt', 'log', 'ones'}
            Weightning policy for the field map histogram (default is "full")
        num_interpolators: int, "auto"
            Number of interpolators used for multi-linear correction.
            When "auto", the value is the field map frequency range
            divided by 30.
        n_bins: int, "auto"
            Number of bins for the field map histogram.
            When "auto", the value is the field map frequency range
            multiplied by 2.
        """

        # Redirect fourier_op essential variables
        self.fourier_op = fourier_op
        self.shape = fourier_op.shape
        self.samples = fourier_op.samples
        self.n_coils = fourier_op.n_coils

        # Initialize default values
        range_w = (np.min(field_map), np.max(field_map))
        if (num_interpolators == "auto"):
            num_interpolators = int(np.around((range_w[1] - range_w[0]) / 30))
        if (n_bins == "auto"):
            n_bins = 2 * int(np.around(range_w[1] - range_w[0]))

        # Initialize wrapper variables
        self.mask = mask
        self.time_vec = time_vec
        self.n_bins = n_bins
        self.num_interpolators = num_interpolators
        self.coefficients = coefficients
        self.weights = weights

        # Prepare indices to reformat C from E=BC
        self.field_map = field_map
        scale = (range_w[1] - range_w[0]) / self.n_bins
        scale = scale if (scale != 0) else 1
        self.indices = np.around((field_map - range_w[0]) / scale).astype(int)
        self.indices = np.clip(self.indices, 0, self.n_bins - 1)

        # Compute the E=BC factorization and reformat B
        self.B, self.C, self.E = compute_orc_coefficients(
            field_map,
            time_vec,
            mask,
            coefficients,
            num_interpolators,
            weights,
            n_bins
        )

        # Prepare B to match fourier.op shape
        if (hasattr(fourier_op, "mask")):
            self.B = np.tile(
                self.B,
                (fourier_op.mask.size // self.B.shape[0], 1)
            )
            self.B = self.B.reshape((*(fourier_op.mask.shape),
                                     self.num_interpolators))
        else:
            self.B = np.tile(
                self.B,
                (self.samples.shape[0] // self.B.shape[0], 1)
            )

        # Force cast large variables into numpy.complex64
        self.B = self.B.astype(np.complex64)
        self.C = self.C.astype(np.complex64)
        self.E = self.E.astype(np.complex64)

    def op(self, x, *args):
        """
        This method calculates a distorded masked Fourier
        transform of a N-D volume.

        Parameters
        ----------
        x: numpy.ndarray
            input N-D array with the same shape as fourier_op.shape
        Returns
        -------
            masked distorded Fourier transform of the input volume
        """
        y = 0
        for l in range(self.num_interpolators):
            y += self.B[..., l] * self.fourier_op.op(
                self.C[l, self.indices] * x,
                *args
            )
        return y

    def adj_op(self, x, *args):
        """
        This method calculates an inverse masked Fourier
        transform of a distorded N-D k-space.

        Parameters
        ----------
        x: numpy.ndarray
            masked distorded N-D k-space
        Returns
        -------
            inverse Fourier transform of the distorded input k-space.
        """
        y = 0
        for l in range(self.num_interpolators):
            y += np.conj(self.C[l, self.indices]) * self.fourier_op.adj_op(
                np.conj(self.B[..., l]) * x,
                *args
            )
        return y
