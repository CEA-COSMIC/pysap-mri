# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This implements calibrationless reconstruction with different proximities
"""
# System import
import warnings

from .base import ReconstructorBase
from ..operators import GradAnalysis, GradSynthesis, WaveletN

# Third party import
from modopt.opt.proximity import SparseThreshold
from modopt.opt.linear import Identity


class CalibrationlessReconstructor(ReconstructorBase):
    """ This class implements a regularized calibrationless reconstruction.

    Notes
    -----
        For the Analysis case, finds the solution for x of:
        ..math:: (1/2) * sum(||F x_l - y_l||^2_2, n_coils) +
        mu * H(W x_l)

        For the Synthesis case, finds the solution of:
        ..math:: (1/2) * sum(||F Wt alpha_l - y_l||^2_2, n_coils) +
        mu * H(alpha_l)

    Parameters
    ----------
    fourier_op: object of class FFT, NonCartesianFFT or Stacked3DNFFT in
    mri.operators
        Defines the fourier operator F in the above equation.
    linear_op: object, (optional, default None)
        Defines the linear sparsifying operator W. This must operate on x and
        have 2 functions, op(x) and adj_op(coeff) which implements the
        operator and adjoint operator. For wavelets, this can be object of
        class WaveletN or WaveletUD2 from mri.operators .
        If None, sym8 wavelet with nb_scale=3 is chosen.
    gradient_formulation: str between 'analysis' or 'synthesis',
        default 'synthesis'
        defines the formulation of the image model which defines the gradient.
    n_jobs : int, default 1
        The number of cores to be used for faster reconstruction
    verbose: int, optional default 0
        Verbosity levels
            1 => Print basic debug information
            5 => Print all initialization information
            20 => Calculate cost at the end of each iteration.
            30 => Print the debug information of operators if defined by class
            NOTE - High verbosity (>20) levels are computationally intensive.
    **kwargs : Extra keyword arguments
        for gradient initialization:
            Please refer to mri.operators.gradient.base for information
        regularizer_op: operator, (optional default None)
            Defines the regularization operator for the regularization
            function H. If None, the  regularization chosen is Identity and
            the optimization turns to gradient descent.
    """

    def __init__(self, fourier_op, linear_op=None,
                 gradient_formulation="synthesis", n_jobs=1, verbose=0,
                 **kwargs):
        if linear_op is None:
            linear_op = WaveletN(
                # TODO change nb_scales to max_nb_scale - 1
                wavelet_name="sym8",
                nb_scale=3,
                dim=len(fourier_op.shape),
                n_coils=fourier_op.n_coils,
                n_jobs=n_jobs,
                verbose=bool(verbose >= 30),
            )
        # Ensure that we are in right multichannel config
        if fourier_op.n_coils != linear_op.n_coils:
            raise ValueError("The value of n_coils for fourier and wavelet "
                             "operation must be same for "
                             "calibrationless reconstruction!")
        if gradient_formulation == 'analysis':
            grad_class = GradAnalysis
        elif gradient_formulation == 'synthesis':
            grad_class = GradSynthesis
        super(CalibrationlessReconstructor, self).__init__(
            fourier_op=fourier_op,
            linear_op=linear_op,
            gradient_formulation=gradient_formulation,
            grad_class=grad_class,
            verbose=verbose,
            **kwargs,
        )
