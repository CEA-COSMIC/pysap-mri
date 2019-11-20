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
    For the Analysis case finds the solution  for x of:
        (1/2) * sum(||F x - y||^2_2, 1) + mu * ||Wt x||_1

    For the Synthesise case finds the solution of:
        (1/2) * sum(||F Wt alpha - y||^2_2, 1) + mu * ||alpha||_1

    Parameters
    ----------
    fourier_op: object of class FFT, NonCartesianFFT or Stacked3DNFFT in
                mri.operators
        Defines the fourier operator F in the above equation.
    linear_op: object, (optional, default None)
        Defines the linear sparsifying operator Wt. This must operate on x and
        have 2 functions, op(x) and adj_op(coeff) which implements the
        operator and adjoint operator. For wavelets, this can be object of
        class WaveletN or WaveletUD2 from mri.operators .
        If None, sym8 wavelet with nb_scales=3 is chosen.
    prox_op: operator, (optional default None)
        Defines the proximity operator for the regularization function.
        For example, for L1 Norm, the proximity operator is Thresholding
        If None, the proximity opertaor is defined as Soft thresholding
        of wavelet coefficients with mu value as specified.
    mu: float, (optional, default 0)
        If prox_op is None, the value of mu is used to form a proximity
        operator that is soft thresholding of the wavelet coefficients.
    lips_calc_max_iter: int, default 10
        Defines the maximum number of iterations to calculate the lipchitz
        constant
    num_check_lips: int, default 10
        Number of iterations to check if the lipchitz constant is correct
    optimization_alg: str, default 'pogm'
        Type of optimization algorithm to use, 'pogm' | 'fista' | 'condatvu'
    lipschitz_cst: int, default None
        The user specified lipschitz constant. If this is not specified,
        it is calculated using PowerMethod
    verbose: int, default 0
        Verbosity level.
            1 => Print basic debug information
            5 => Print all initialization information
            20 => Calculate cost at the end of each iteration.
                NOTE : This is computationally intensive.
            30 => Print the debug information of operators if defined by class
    Note:
    -----
    The user is expected to specify the either prox_op or mu to obtain
    reconstructions, else the above equations lose the regularization terms
    resulting in inverse transform as solution.
    The reconstruction in this case proceeds with a warning.
    """

    def __init__(self, fourier_op, linear_op=None, prox_op=None, mu=0,
                 gradient_method="synthesis", lips_calc_max_iter=10,
                 num_check_lips=10, lipschitz_cst=None,
                 optimization_alg='pogm', verbose=0):
        # Ensure that we are not in multichannel config
        if linear_op is None:
            linear_op = WaveletN(
                wavelet_name="sym8",
                dim=len(fourier_op.shape),
                nb_scale=3,
                verbose=bool(verbose >= 30),
            )
        if fourier_op.n_coils != 1 or linear_op.n_coils != 1:
            raise ValueError("The value of n_coils cannot be greater than 1 "
                             "for single channel reconstruction")
        if gradient_method == 'analysis':
            grad_class = GradAnalysis
        elif gradient_method == 'synthesis':
            grad_class = GradSynthesis
        super(SingleChannelReconstructor, self).__init__(
            fourier_op=fourier_op,
            linear_op=linear_op,
            prox_op=prox_op,
            mu=mu,
            gradient_method=gradient_method,
            grad_class=grad_class,
            lipschitz_cst=lipschitz_cst,
            lips_calc_max_iter=lips_calc_max_iter,
            num_check_lips=num_check_lips,
            optimization_alg=optimization_alg,
            verbose=verbose,
        )