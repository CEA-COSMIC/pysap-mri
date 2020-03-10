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
from modopt.opt.linear import Identity


class ReconstructorBase(object):
    """ This is the base reconstructor class for reconstruction.
    This class holds some parameters that are common for all MR Image
    reconstructors.

    Notes
    -----
        For the Analysis case, finds the solution  for x of:
        ..math:: (1/2) * ||F x - y||^2_2 + mu * H (W x)

        For the Synthesis case, finds the solution of:
        ..math:: (1/2) * ||F Wt alpha - y||^2_2 + mu * H(alpha)

    Parameters
    ----------
    fourier_op: object of class FFT, NonCartesianFFT or Stacked3DNFFT in
    mri.operators
        Defines the fourier operator F.
    linear_op: object
        Defines the linear sparsifying operator W. This must operate on x and
        have 2 functions, op(x) and adj_op(coeff) which implements the
        operator and adjoint operator. For wavelets, this can be object of
        class WaveletN or WaveletUD2 from mri.operators
    regularizer_op: operator, (optional default None)
        Defines the regularization operator for the regularization function H.
        If None, the  regularization chosen is Identity and the optimization
        turns to gradient descent.
    gradient_formulation: str between 'analysis' or 'synthesis',
        default 'synthesis'
        defines the formulation of the image model which defines the gradient.
    grad_class: Gradient class from mri.operators.
        Points to the gradient class based on the MR Image model and
        gradient_formulation.
    init_gradient_op: bool, default True
        This parameter controls whether the gradient operator must be
        initialized right now.
        If set to false, the user needs to call initialize_gradient_op to
        initialize the gradient at right time before reconstruction
    verbose: int, optional default 0
        Verbosity levels
            1 => Print basic debug information
            5 => Print all initialization information
            20 => Calculate cost at the end of each iteration.
            30 => Print the debug information of operators if defined by class
            NOTE - High verbosity (>20) levels are computationally intensive.
    extra_grad_args: Extra Keyword arguments for gradient initialization
        This holds the initialization parameters used for gradient
        initialization which is obtained from 'grad_class'.
        Please refer to mri.operators.gradient.base for reference.
        In case of sythesis formulation, the 'linear_op' is also passed as
        an extra arg
    """

    def __init__(self, fourier_op, linear_op, regularizer_op,
                 gradient_formulation, grad_class, init_gradient_op=True,
                 verbose=0, **extra_grad_args):
        self.fourier_op = fourier_op
        self.linear_op = linear_op
        self.prox_op = regularizer_op
        self.gradient_method = gradient_formulation
        self.grad_class = grad_class
        self.verbose = verbose
        self.extra_grad_args = extra_grad_args
        if regularizer_op is None:
            warnings.warn("The prox_op is not set. Setting to identity. "
                          "Note that optimization is just a gradient descent.")
            self.prox_op = Identity()
        # TODO try to not use gradient_formulation and
        #  rely on static attributes
        # If the reconstruction formulation is synthesis,
        # we send the linear operator as well.
        if gradient_formulation == 'synthesis':
            self.extra_grad_args['linear_op'] = self.linear_op
        if init_gradient_op:
            self.initialize_gradient_op(**self.extra_grad_args)

    def initialize_gradient_op(self, **extra_args):
        # Initialize gradient operator and cost operators
        self.gradient_op = self.grad_class(
            fourier_op=self.fourier_op,
            verbose=self.verbose,
            **extra_args,
        )

    def reconstruct(self, kspace_data, optimization_alg='pogm',
                    x_init=None, num_iterations=100, cost_op_kwargs=None,
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
            input initial guess image for reconstruction. If None, the
            initialization will be zero
        num_iterations: int (optional, default 100)
            number of iterations of algorithm
        cost_op_kwargs: dict (optional, default None)
            specifies the extra keyword arguments for cost operations.
            please refer to modopt.opt.cost.costObj for details.
        kwargs: extra keyword arguments for modopt algorithm
            Please refer to corresponding ModOpt algorithm class for details.
            https://github.com/CEA-COSMIC/ModOpt/blob/master/\
            modopt/opt/algorithms.py
        """
        self.gradient_op.obs_data = kspace_data
        available_algorithms = ["condatvu", "fista", "pogm"]
        if optimization_alg not in available_algorithms:
            raise ValueError("The optimization_alg must be one of " +
                             str(available_algorithms))
        optimizer = eval(optimization_alg)
        if optimization_alg == "condatvu":
            kwargs["dual_regularizer"] = self.prox_op
            optimizer_type = 'primal_dual'
        else:
            kwargs["prox_op"] = self.prox_op
            optimizer_type = 'forward_backward'
        if cost_op_kwargs is None:
            cost_op_kwargs = {}
        self.cost_op = GenericCost(
            gradient_op=self.gradient_op,
            prox_op=self.prox_op,
            verbose=self.verbose >= 20,
            optimizer_type=optimizer_type,
            **cost_op_kwargs,
        )
        self.x_final, self.costs, *metrics = optimizer(
                gradient_op=self.gradient_op,
                linear_op=self.linear_op,
                cost_op=self.cost_op,
                max_nb_of_iter=num_iterations,
                x_init=x_init,
                verbose=self.verbose,
                **kwargs)
        if optimization_alg == 'condatvu':
            self.metrics, self.y_final = metrics
        else:
            self.metrics = metrics[0]
        return self.x_final, self.costs, self.metrics
