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
    """ This class implements a calibrationless reconstruction based on the
    L1-norm regularization.
    For the Analysis case finds the solution for x of:
        (1/2) * sum(||F x_l - y_l||^2_2, n_coils) +
                    mu * H(Wt x_l)
    For the Synthesis case finds the solution of:
        (1/2) * sum(||F Wt alpha_l - y_l||^2_2, n_coils) +
                    mu * H(alpha_l)
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
        Defines the proximity operator for the regularization function H.
        For example, for L1 Norm, the proximity operator is Thresholding
        If None, the proximity opertaor is defined as Soft thresholding
        of wavelet coefficients with mu value as specified.
    mu: float or np.ndarray, (optional, default 0)
        If prox_op is None, the value of mu is used to form a proximity
        operator that is soft thresholding of the wavelet coefficients.
        If prox_op is specified, this is ignored.
    gradient_formulation: str between 'analysis' or 'synthesis',
        default 'synthesis'
        defines the formulation of the image model which defines the gradient.
    lips_calc_max_iter: int, default 10
        Defines the maximum number of iterations to calculate the lipchitz
        constant
    num_check_lips: int, default 10
        Number of iterations to check if the lipchitz constant is correct
    lipschitz_cst: int, default None
        The user specified lipschitz constant. If this is not specified,
        it is calculated using PowerMethod
    n_jobs : int, default 1
        The number of cores to be used for faster reconstruction
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
                 gradient_formulation="synthesis", lips_calc_max_iter=10,
                 num_check_lips=10, lipschitz_cst=None, n_jobs=1, verbose=0):
        if linear_op is None:
            linear_op = WaveletN(
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
            prox_op=prox_op,
            mu=mu,
            gradient_formulation=gradient_formulation,
            grad_class=grad_class,
            lipschitz_cst=lipschitz_cst,
            lips_calc_max_iter=lips_calc_max_iter,
            num_check_lips=num_check_lips,
            verbose=verbose,
        )
