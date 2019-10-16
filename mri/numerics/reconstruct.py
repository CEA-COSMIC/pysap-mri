# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
FISTA or CONDAT-VU MRI reconstruction.
"""


# System import
from __future__ import print_function
import copy
import time
import warnings

# Package import
from mri.numerics.reweight import mReweight
from pysap.utils import fista_logo
from pysap.utils import condatvu_logo
from pysap.base.utils import unflatten

# Third party import
import numpy as np
from modopt.math.stats import sigma_mad
from modopt.opt.linear import Identity
from modopt.opt.proximity import Positivity
from modopt.opt.algorithms import Condat, ForwardBackward, POGM
from modopt.opt.reweight import cwbReweight


def sparse_rec_fista(gradient_op, linear_op, prox_op, cost_op,
                     lambda_init=1.0, max_nb_of_iter=300,
                     metric_call_period=5, metrics={},
                     verbose=0, **lambda_update_params):
    """ The FISTA sparse reconstruction without reweightings.

    .. note:: At the moment, tested only with 2D data.

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
    start = time.clock()

    # Define the initial primal and dual solutions
    x_init = np.zeros(gradient_op.fourier_op.shape, dtype=np.complex)
    alpha = linear_op.op(x_init)
    alpha[...] = 0.0

    # Welcome message
    if verbose > 0:
        print(fista_logo())
        print(" - mu: ", prox_op.weights)
        print(" - lipschitz constant: ", gradient_op.spec_rad)
        print(" - data: ", gradient_op.fourier_op.shape)
        if hasattr(linear_op, "nb_scale"):
            print(" - wavelet: ", linear_op, "-", linear_op.nb_scale)
        print(" - max iterations: ", max_nb_of_iter)
        print(" - image variable shape: ", x_init.shape)
        print(" - alpha variable shape: ", alpha.shape)
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
        x=alpha,
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
    end = time.clock()
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


def sparse_rec_condatvu(gradient_op, linear_op, prox_dual_op, cost_op,
                        std_est=None, std_est_method=None, std_thr=2.,
                        tau=None, sigma=None, relaxation_factor=1.0,
                        nb_of_reweights=1, max_nb_of_iter=150,
                        add_positivity=False, atol=1e-4, metric_call_period=5,
                        metrics={}, verbose=0):
    """ The Condat-Vu sparse reconstruction with reweightings.

    .. note:: At the moment, tested only with 2D data.

    Parameters
    ----------
    gradient_op: instance of class GradBase
        the gradient operator.
    linear_op: instance of LinearBase
        the linear operator: seek the sparsity, ie. a wavelet transform.
    prox_dual_op: instance of ProximityParent
        the proximal dual operator.
    cost_op: instance of costObj
        the cost function used to check for convergence during the
        optimization.
    std_est: float, default None
        the noise std estimate.
        If None use the MAD as a consistent estimator for the std.
    std_est_method: str, default None
        if the standard deviation is not set, estimate this parameter using
        the mad routine in the image ('primal') or in the sparse wavelet
        decomposition ('dual') domain.
    std_thr: float, default 2.
        use this treshold expressed as a number of sigma in the residual
        proximity operator during the thresholding.
    tau, sigma: float, default None
        parameters of the Condat-Vu proximal-dual splitting algorithm.
        If None estimates these parameters.
    relaxation_factor: float, default 0.5
        parameter of the Condat-Vu proximal-dual splitting algorithm.
        If 1, no relaxation.
    nb_of_reweights: int, default 1
        the number of reweightings.
    max_nb_of_iter: int, default 150
        the maximum number of iterations in the Condat-Vu proximal-dual
        splitting algorithm.
    add_positivity: bool, default False
        by setting this option, set the proximity operator to identity or
        positive.
    atol: float, default 1e-4
        tolerance threshold for convergence.
    metric_call_period: int (default 5)
        the period on which the metrics are compute.
    metrics: dict (optional, default None)
        the list of desired convergence metrics: {'metric_name':
        [@metric, metric_parameter]}. See modopt for the metrics API.
    verbose: int, default 0
        the verbosity level.

    Returns
    -------
    x_final: ndarray
        the estimated CONDAT-VU solution.
    costs: list of float
        the cost function values.
    metrics: dict
        the requested metrics values during the optimization.
    y_final: ndarrat
        the estimated dual CONDAT-VU solution
    """
    # Check inputs
    start = time.clock()
    if std_est_method not in (None, "primal", "dual"):
        raise ValueError(
            "Unrecognize std estimation method '{0}'.".format(std_est_method))

    # Define the initial primal and dual solutions
    x_init = np.zeros(gradient_op.fourier_op.shape, dtype=np.complex)
    weights = linear_op.op(x_init)

    # Define the weights used during the thresholding in the dual domain,
    # the reweighting strategy, and the prox dual operator

    # Case1: estimate the noise std in the image domain
    if std_est_method == "primal":
        if std_est is None:
            std_est = sigma_mad(gradient_op.MtX(data))
        weights[...] = std_thr * std_est
        reweight_op = cwbReweight(weights)
        prox_dual_op.weights = reweight_op.weights

    # Case2: estimate the noise std in the sparse wavelet domain
    elif std_est_method == "dual":
        if std_est is None:
            std_est = 0.0
        weights[...] = std_thr * std_est
        reweight_op = mReweight(weights, linear_op, thresh_factor=std_thr)
        prox_dual_op.weights = reweight_op.weights

    # Case3: manual regularization mode, no reweighting
    else:
        reweight_op = None
        nb_of_reweights = 0

    # Define the Condat Vu optimizer: define the tau and sigma in the
    # Condat-Vu proximal-dual splitting algorithm if not already provided.
    # Check also that the combination of values will lead to convergence.
    norm = linear_op.l2norm(gradient_op.fourier_op.shape)
    lipschitz_cst = gradient_op.spec_rad
    if sigma is None:
        sigma = 0.5
    if tau is None:
        # to avoid numerics troubles with the convergence bound
        eps = 1.0e-8
        # due to the convergence bound
        tau = 1.0 / (lipschitz_cst/2 + sigma * norm**2 + eps)
    convergence_test = (
        1.0 / tau - sigma * norm ** 2 >= lipschitz_cst / 2.0)

    # Define initial primal and dual solutions
    primal = np.zeros(gradient_op.fourier_op.shape, dtype=np.complex)
    dual = linear_op.op(primal)
    dual[...] = 0.0

    # Welcome message
    if verbose > 0:
        print(condatvu_logo())
        print(" - mu: ", prox_dual_op.weights)
        print(" - lipschitz constant: ", gradient_op.spec_rad)
        print(" - tau: ", tau)
        print(" - sigma: ", sigma)
        print(" - rho: ", relaxation_factor)
        print(" - std: ", std_est)
        print(" - 1/tau - sigma||L||^2 >= beta/2: ", convergence_test)
        print(" - data: ", gradient_op.fourier_op.shape)
        if hasattr(linear_op, "nb_scale"):
            print(" - wavelet: ", linear_op, "-", linear_op.nb_scale)
        print(" - max iterations: ", max_nb_of_iter)
        print(" - number of reweights: ", nb_of_reweights)
        print(" - primal variable shape: ", primal.shape)
        print(" - dual variable shape: ", dual.shape)
        print("-" * 40)

    # Define the proximity operator
    if add_positivity:
        prox_op = Positivity()
    else:
        prox_op = Identity()

    # Define the optimizer
    opt = Condat(
        x=primal,
        y=dual,
        grad=gradient_op,
        prox=prox_op,
        prox_dual=prox_dual_op,
        linear=linear_op,
        cost=cost_op,
        rho=relaxation_factor,
        sigma=sigma,
        tau=tau,
        rho_update=None,
        sigma_update=None,
        tau_update=None,
        auto_iterate=False,
        metric_call_period=metric_call_period,
        metrics=metrics)
    cost_op = opt._cost_func

    # Perform the first reconstruction
    if verbose > 0:
        print("Starting optimization...")
    opt.iterate(max_iter=max_nb_of_iter)

    # Loop through the number of reweightings
    for reweight_index in range(nb_of_reweights):

        # Generate the new weights following reweighting prescription
        if std_est_method == "primal":
            reweight_op.reweight(linear_op.op(opt._x_new))
        else:
            std_est = reweight_op.reweight(opt._x_new)

        # Welcome message
        if verbose > 0:
            print(" - reweight: ", reweight_index + 1)
            print(" - std: ", std_est)

        # Update the weights in the dual proximity operator
        prox_dual_op.weights = reweight_op.weights

        # Perform optimisation with new weights
        opt.iterate(max_iter=max_nb_of_iter)

    # Goodbye message
    end = time.clock()
    if verbose > 0:
        if hasattr(cost_op, "cost"):
            print(" - final iteration number: ", cost_op._iteration)
            print(" - final cost value: ", cost_op.cost)
        print(" - converged: ", opt.converge)
        print("Done.")
        print("Execution time: ", end - start, " seconds")
        print("-" * 40)

    # Get the final solution
    x_final = opt.x_final
    y_final = opt.y_final
    if hasattr(cost_op, "cost"):
        costs = cost_op._cost_list
    else:
        costs = None

    return x_final, costs, opt.metrics, y_final


def sparse_rec_pogm(gradient_op, linear_op, prox_op, cost_op=None,
                    max_nb_of_iter=300, metric_call_period=5, sigma_bar=0.96,
                    metrics={}, verbose=0):
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
    start = time.clock()

    # Define the initial values
    im_shape = gradient_op.fourier_op.shape
    zeros_right_shape = linear_op.op(np.zeros(im_shape, dtype='complex128'))

    # Welcome message
    if verbose > 0:
        # TODO: think of logo for POGM
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
        u=zeros_right_shape,
        x=zeros_right_shape,
        y=zeros_right_shape,
        z=zeros_right_shape,
        grad=gradient_op,
        prox=prox_op,
        cost=cost_op,
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
    end = time.clock()
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
