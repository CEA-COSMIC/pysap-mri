# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This implements the single channel reconstruction.
"""

from .base import ReconstructorBase
from ..operators import GradSynthesis, GradAnalysis, WaveletN


class SingleChannelReconstructor(ReconstructorBase):
    """ This class implements the Single channel MR image Reconstruction.

    Notes
    -----
        For the Analysis case, finds the solution  for x of:
        ..math:: (1/2) * sum(||F x - y||^2_2, 1) + mu * H (W x)

        For the Synthesis case, finds the solution of:
        ..math:: (1/2) * sum(||F Wt alpha - y||^2_2, 1) + mu * H (alpha)

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
    regularizer_op: operator, (optional default None)
        Defines the regularization operator for the regularization function H.
        If None, the  regularization chosen is Identity and the optimization
        turns to gradient descent.
    gradient_formulation: str between 'analysis' or 'synthesis',
        default 'synthesis'
        defines the formulation of the image model which defines the gradient.
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
                 gradient_formulation="synthesis", verbose=0, **kwargs):
        # Ensure that we are not in multichannel config
        if linear_op is None:
            # TODO change nb_scales to max_nb_scale - 1
            linear_op = WaveletN(
                wavelet_name="sym8",
                dim=len(fourier_op.shape),
                nb_scale=3,
                verbose=bool(verbose >= 30),
            )
        if fourier_op.n_coils != 1 or linear_op.n_coils != 1:
            raise ValueError("The value of n_coils cannot be greater than 1 "
                             "for single channel reconstruction")
        if gradient_formulation == 'analysis':
            grad_class = GradAnalysis
        elif gradient_formulation == 'synthesis':
            grad_class = GradSynthesis
        super(SingleChannelReconstructor, self).__init__(
            fourier_op=fourier_op,
            linear_op=linear_op,
            gradient_formulation=gradient_formulation,
            grad_class=grad_class,
            verbose=verbose,
            **kwargs,
        )
