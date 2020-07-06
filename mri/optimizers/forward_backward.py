# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
FISTA or POGM MRI reconstruction.
"""


# System import
import time

# Third party import
import numpy as np
from modopt.opt.algorithms import ForwardBackward, POGM


def fista(gradient_op, linear_op, prox_op, cost_op,
          lambda_init=1.0, max_nb_of_iter=300, x_init=None,
          metric_call_period=5, metrics={},
          verbose=0, **lambda_update_params):
    """ The FISTA sparse reconstruction

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
    lambda_init: float, (default 1.0)
        initial value for the FISTA step.
    max_nb_of_iter: int (optional, default 300)
        the maximum number of iterations in the Condat-Vu proximal-dual
        splitting algorithm.
    x_init: np.ndarray (optional, default None)
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
    x_final: ndarray
        the estimated FISTA solution.
    costs: list of float
        the cost function values.
    metrics: dict
        the requested metrics values during the optimization.
    """
    start = time.perf_counter()

    # Define the initial primal and dual solutions
    if x_init is None:
        x_init = np.squeeze(np.zeros((gradient_op.linear_op.n_coils,
                                      *gradient_op.fourier_op.shape),
                                     dtype=np.complex))
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
    cost_op = opt._cost_func

    # Perform the reconstruction
    if verbose > 0:
        print("Starting optimization...")
    opt.iterate(max_iter=max_nb_of_iter)
    end = time.perf_counter()
    if verbose > 0:
        # cost_op.plot_cost()
        if hasattr(cost_op, "cost"):
            print(" - final iteration number: ", cost_op._iteration)
            print(" - final log10 cost value: ", np.log10(cost_op.cost))
        print(" - converged: ", opt.converge)
        print("Done.")
        print("Execution time: ", end - start, " seconds")
        print("-" * 40)
    x_final = linear_op.adj_op(opt.x_final)
    if hasattr(cost_op, "cost"):
        costs = cost_op._cost_list
    else:
        costs = None

    return x_final, costs, opt.metrics


def pogm(gradient_op, linear_op, prox_op, cost_op=None,
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
    lambda_init: float, (default 1.0)
        initial value for the FISTA step.
    max_nb_of_iter: int (optional, default 300)
        the maximum number of iterations in the POGM algorithm.
    x_init: np.ndarray (optional, default None)
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
    x_final: ndarray
        the estimated POGM solution.
    costs: list of float
        the cost function values.
    metrics: dict
        the requested metrics values during the optimization.
    """
    start = time.perf_counter()

    # Define the initial values
    im_shape = (gradient_op.linear_op.n_coils, *gradient_op.fourier_op.shape)
    if x_init is None:
        alpha_init = linear_op.op(np.squeeze(np.zeros(im_shape,
                                                      dtype='complex128')))
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

    # Perform the reconstruction
    if verbose > 0:
        print("Starting optimization...")
    opt.iterate(max_iter=max_nb_of_iter)
    end = time.perf_counter()
    if verbose > 0:
        # cost_op.plot_cost()
        if hasattr(cost_op, "cost"):
            print(" - final iteration number: ", cost_op._iteration)
            print(" - final log10 cost value: ", np.log10(cost_op.cost))
        print(" - converged: ", opt.converge)
        print("Done.")
        print("Execution time: ", end - start, " seconds")
        print("-" * 40)
    x_final = linear_op.adj_op(opt.x_final)
    metrics = opt.metrics

    if hasattr(cost_op, "cost"):
        costs = cost_op._cost_list
    else:
        costs = None

    return x_final, costs, metrics
