# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""FISTA or POGM MRI reconstruction."""

# Third party import
import numpy as np
from modopt.opt.algorithms import ForwardBackward, POGM

from .base import run_algorithm, run_online_algorithm


def fista(gradient_op, linear_op, prox_op, cost_op, kspace_generator=None, estimate_call_period=None,
          lambda_init=1.0, max_nb_of_iter=300, x_init=None,
          metric_call_period=5, metrics={},
          verbose=0, **lambda_update_params):
    """FISTA sparse reconstruction.

    Parameters
    ----------
    gradient_op: instance of class GradBase
        the gradient operator.
    linear_op: instance of LinearBase
        the linear operator: seek the sparsity, ie. a wavelet transform.
    prox_op: instance of ProximityParent
        the proximal operator.
    cost_op: instance of costObj
        the cost function used to check for convergence during the
        optimization.
    kspace_generator: instance of class BaseKspaceGenerator, default None
        If not None, run the algorithm in an online way, where the data is
        updated between iterations.
    estimate_call_period: int, default None
        In an online configuration (kspace_generator is defined),
        retrieve partial results at this interval.
    lambda_init: float, (default 1.0)
        initial value for the FISTA step.
    max_nb_of_iter: int (optional, default 300)
        the maximum number of iterations in the Condat-Vu proximal-dual
        splitting algorithm.
    x_init: numpy.ndarray (optional, default None)
        Inital guess for the image
    metric_call_period: int (default 5)
        the period on which the metrics are compute.
    metrics: dict (optional, default None)
        the list of desired convergence metrics: {'metric_name':
        [@metric, metric_parameter]}. See modopt for the metrics API.
    verbose: int (optional, default 0)
        the verbosity level.
    lambda_update_params: dict,
        Parameters for the lambda update in FISTA mode

    Returns
    -------
    x_final: numpy.ndarray
        the estimated FISTA solution.
    costs: list of float
        the cost function values.
    metrics: dict
        the requested metrics values during the optimization.
    """
    # Define the initial primal and dual solutions
    if x_init is None:
        x_init = np.squeeze(np.zeros((gradient_op.linear_op.n_coils,
                                      *gradient_op.fourier_op.shape),
                                     dtype=np.complex64))
    alpha_init = linear_op.op(x_init)

    # Welcome message
    if verbose > 0:
        print(" - mu: ", prox_op.weights)
        print(" - lipschitz constant: ", gradient_op.spec_rad)
        print(" - data: ", gradient_op.fourier_op.shape)
        if hasattr(linear_op, "nb_scale"):
            print(" - wavelet: ", linear_op, "-", linear_op.nb_scale)
        print(" - max iterations: ", max_nb_of_iter)
        print(" - image variable shape: ", gradient_op.fourier_op.shape)
        print(" - alpha variable shape: ", alpha_init.shape)
        print("-" * 40)

    beta_param = gradient_op.inv_spec_rad
    if lambda_update_params.get("restart_strategy") == "greedy":
        lambda_update_params["min_beta"] = gradient_op.inv_spec_rad
        # this value is the recommended one by J. Liang in his article
        # when introducing greedy FISTA.
        # ref: https://arxiv.org/pdf/1807.04005.pdf
        beta_param *= 1.3

    # Define the optimizer
    opt = ForwardBackward(
        x=alpha_init,
        grad=gradient_op,
        prox=prox_op,
        cost=cost_op,
        auto_iterate=False,
        metric_call_period=metric_call_period,
        metrics=metrics,
        linear=linear_op,
        lambda_param=lambda_init,
        beta_param=beta_param,
        **lambda_update_params)
    if kspace_generator is not None:
        return run_online_algorithm(opt, kspace_generator, estimate_call_period, verbose)
    return run_algorithm(opt, max_nb_of_iter, verbose)


def pogm(gradient_op, linear_op, prox_op, cost_op=None, kspace_generator=None, estimate_call_period=None,
         max_nb_of_iter=300, x_init=None, metric_call_period=5,
         sigma_bar=0.96, metrics={}, verbose=0):
    """
    Perform sparse reconstruction using the POGM algorithm.

    Parameters
    ----------
    gradient_op: instance of class GradBase
        the gradient operator.
    linear_op: instance of LinearBase
        the linear operator: seek the sparsity, ie. a wavelet transform.
    prox_op: instance of ProximityParent
        the proximal operator.
    cost_op: instance of costObj, (default None)
        the cost function used to check for convergence during the
        optimization.
    kspace_generator: instance of BaseKspaceGenerator, default None
        If not None, use it to perform an online reconstruction.
    estimate_call_period: int, default None
        In an online configuration (kspace_generator is defined),
        retrieve partial results at this interval.
    lambda_init: float, (default 1.0)
        initial value for the FISTA step.
    max_nb_of_iter: int (optional, default 300)
        the maximum number of iterations in the POGM algorithm.
    x_init: numpy.ndarray (optional, default None)
        the initial guess of image
    metric_call_period: int (default 5)
        the period on which the metrics are computed.
    metrics: dict (optional, default None)
        the list of desired convergence metrics: {'metric_name':
        [@metric, metric_parameter]}. See modopt for the metrics API.
    verbose: int (optional, default 0)
        the verbosity level.

    Returns
    -------
    x_final: numpy.ndarray
        the estimated POGM solution.
    costs: list of float
        the cost function values.
    metrics: dict
        the requested metrics values during the optimization.
    """
    # Define the initial values
    im_shape = (gradient_op.linear_op.n_coils, *gradient_op.fourier_op.shape)
    if x_init is None:
        alpha_init = linear_op.op(np.squeeze(np.zeros(im_shape,
                                                      dtype=np.complex64)))
    else:
        alpha_init = linear_op.op(x_init)

    # Welcome message
    if verbose > 0:
        print(" - mu: ", prox_op.weights)
        print(" - lipschitz constant: ", gradient_op.spec_rad)
        print(" - data: ", gradient_op.fourier_op.shape)
        if hasattr(linear_op, "nb_scale"):
            print(" - wavelet: ", linear_op, "-", linear_op.nb_scale)
        print(" - max iterations: ", max_nb_of_iter)
        print(" - image variable shape: ", im_shape)
        print("-" * 40)

    # Hyper-parameters
    beta = gradient_op.inv_spec_rad

    opt = POGM(
        u=alpha_init,
        x=alpha_init,
        y=alpha_init,
        z=alpha_init,
        grad=gradient_op,
        prox=prox_op,
        cost=cost_op,
        linear=linear_op,
        beta_param=beta,
        sigma_bar=sigma_bar,
        metric_call_period=metric_call_period,
        metrics=metrics,
        auto_iterate=False,
    )
    if kspace_generator is not None:
        return run_online_algorithm(opt, kspace_generator, estimate_call_period, verbose)
    return run_algorithm(opt, max_nb_of_iter, verbose=verbose)
