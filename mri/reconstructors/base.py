# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import warnings

# Package import
from ..operators.linear.wavelet import WaveletUD2, WaveletN
from ..optimizers import pogm, condatvu, fista
from ..optimizers.utils.cost import GenericCost

# Third party import
from modopt.opt.proximity import SparseThreshold
from modopt.opt.linear import Identity


class ReconstructorBase(object):
    """ This is the base reconstructor class for reconstruction.
    This class holds some parameters that is common for all MR Image
    reconstructors
    For the Analysis case finds the solution  for x of:
        (1/2) * sum(||F x - y||^2_2, 1) + mu * ||Wt x||_1

    For the Synthesise case finds the solution of:
        (1/2) * sum(||F Wt alpha - y||^2_2, 1) + mu * ||alpha||_1
    Parameters
    ----------
    fourier_op: object of class FFT, NonCartesianFFT or Stacked3DNFFT in
                mri.operators
        Defines the fourier operator F.
    linear_op: object
        Defines the linear sparsifying operator Wt. This must operate on x and
        have 2 functions, op(x) and adj_op(coeff) which implements the
        operator and adjoint operator. For wavelets, this can be object of
        class WaveletN or WaveletUD2 from mri.operators .
    prox_op: object that implements proximal operator
        Defines the proximity operator for the regularization function.
        For example, for L1 Norm, the proximity operator is Thresholding
    mu: float or np.ndarray
        If prox_op is None, the value of mu is used to form a proximity
        operator that is soft thresholding of the wavelet coefficients.
        If prox_op is specified, this is ignored.
    lips_calc_max_iter: int
        Defines the maximum number of iterations to calculate the lipchitz
        constant
    num_check_lips: int
        Number of iterations to check if the lipchitz constant is correct
    lipschitz_cst: int, default None
        The user specified lipschitz constant. If this is not specified,
        it is calculated using PowerMethod
    verbose: int
        Verbosity level.
            1 => Print basic debug information
            5 => Print all initialization information
            20 => Calculate cost at the end of each iteration.
                NOTE : This is computationally intensive.
            30 => Print the debug information of operators if defined by class
    init_gradient_op: bool, default True
        This parameter controls whether the gradient operator must be
        initialized right now.
        If set to false, the user needs to call initialize_gradient_op to
        initialize the gradient at right time before reconstruction
    Note:
    -----
    The user is expected to specify the either prox_op or mu to obtain
    reconstructions, else the above equations lose the regularization terms
    resulting in inverse transform as solution.
    """

    def __init__(self, fourier_op, linear_op, prox_op, mu, gradient_method,
                 grad_class, lips_calc_max_iter, num_check_lips, lipschitz_cst,
                 verbose, init_gradient_op=True,
                 **extra_grad_args):
        self.fourier_op = fourier_op
        self.linear_op = linear_op
        self.prox_op = prox_op
        self.gradient_method = gradient_method
        self.grad_class = grad_class
        self.lipschitz_cst = lipschitz_cst
        self.lips_calc_max_iter = lips_calc_max_iter
        self.num_check_lips = num_check_lips
        self.verbose = verbose
        self.extra_grad_args = extra_grad_args
        if prox_op is None and mu == 0:
            warnings.warn("The prox_op is not set and mu = 0. The result will "
                          "not be a reconstruction but an inverse")

        # If the reconstruction formulation is synthesis,
        # we send the linear operator as well.
        if gradient_method == 'synthesis':
            self.extra_grad_args['linear_op'] = self.linear_op
        if self.prox_op is None:
            if gradient_method == 'synthesis':
                self.prox_op = SparseThreshold(
                    Identity(),
                    mu,
                    thresh_type="soft",
                )
            elif gradient_method == "analysis":
                if self.prox_op is None:

                    self.prox_op = SparseThreshold(
                        self.linear_op,
                        mu,
                        thresh_type="soft",
                    )
        if init_gradient_op:
            self.initialize_gradient_op(**self.extra_grad_args)

    def initialize_gradient_op(self, **extra_args):
        # Initialize gradient operator and cost operators
        self.gradient_op = self.grad_class(
            fourier_op=self.fourier_op,
            lips_calc_max_iter=self.lips_calc_max_iter,
            lipschitz_cst=self.lipschitz_cst,
            num_check_lips=self.num_check_lips,
            verbose=self.verbose,
            **extra_args,
        )
        self.cost_op = GenericCost(
            gradient_op=self.gradient_op,
            prox_op=self.prox_op,
            verbose=self.verbose >= 20,
        )

    def reconstruct(self, kspace_data, optimization_alg='pogm',
                    x_init=None, num_iterations=100, reinit_grad_op=False,
                    **kwargs):
        """ This method calculates operator transform.
        Parameters
        ----------
        kspace_data: np.ndarray
            the acquired value in the Fourier domain.
            this is y in above equation.
        optimization_alg: str (optional, default 'pogm')
            Type of optimization algorithm to use, 'pogm' | 'fista' |
            'condatvu'
        x_init: np.ndarray (optional, default None)
            input initial guess image for reconstruction
        num_iterations: int (optional, default 100)
            number of iterations of algorithm
        """
        self.gradient_op.obs_data = kspace_data
        if optimization_alg == "fista":
            optimizer = fista
        elif optimization_alg == "condatvu":
            optimizer = condatvu
        elif optimization_alg == "pogm":
            optimizer = pogm
        else:
            raise ValueError("The optimization_alg must be either 'fista' or "
                             "'condatvu or 'pogm'")
        self.x_final, self.costs, *metrics = optimizer(
                gradient_op=self.gradient_op,
                linear_op=self.linear_op,
                prox_op=self.prox_op,
                cost_op=self.cost_op,
                max_nb_of_iter=num_iterations,
                x_init=x_init,
                verbose=self.verbose,
                **kwargs)
        if optimization_alg == 'condatvu':
            self.metrics, self.y_final = metrics
        else:
            self.metrics = metrics
        return self.x_final, self.costs, self.metrics
